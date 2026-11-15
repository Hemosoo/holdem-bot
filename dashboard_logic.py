"""Validated dashboard view models and adapter for the existing hemobot policy."""
import math
import re
from hemo_bot import decide_action
from poker_env import GameState, FOLD, CALL, RAISE

CARD=re.compile(r'^[2-9TJQKA][cdhs]$')


def positions(seats, dealer):
    seats=sorted(set(seats))
    if dealer not in seats or not 2<=len(seats)<=9:
        return {}
    order=seats[seats.index(dealer):]+seats[:seats.index(dealer)]
    labels={2:['BTN / SB','BB'],3:['BTN','SB','BB'],4:['BTN','SB','BB','CO'],
            5:['BTN','SB','BB','UTG','CO'],6:['BTN','SB','BB','UTG','HJ','CO'],
            7:['BTN','SB','BB','UTG','LJ','HJ','CO'],
            8:['BTN','SB','BB','UTG','UTG+1','LJ','HJ','CO'],
            9:['BTN','SB','BB','UTG','UTG+1','UTG+2','LJ','HJ','CO']}[len(order)]
    return dict(zip(order,labels))


def validate_context(context):
    for key,choices in {'turn':('auto','yes','no'),'street':('auto','preflop','flop','turn','river'),
                        'last_to_act':('unknown','yes','no')}.items():
        if key in context and context[key] not in choices: raise ValueError('Invalid '+key)
    if 'hero' in context and context['hero'] is None: raise ValueError('Invalid hero')
    for key in ('hero','dealer'):
        value=context.get(key)
        if value is not None and (type(value) is not int or not 0<=value<8): raise ValueError('Invalid '+key)
    value=context.get('aggressor')
    if value is not None and value not in ('none','unknown') and (type(value) is not int or not 0<=value<8): raise ValueError('Invalid aggressor')
    value=context.get('call')
    if value is not None and (type(value) not in (int,float) or not math.isfinite(value) or value<0): raise ValueError('Amount to call must be nonnegative and finite.')
    value=context.get('raises')
    if value is not None and (type(value) is not int or not 0<=value<=20): raise ValueError('Invalid raise count')
    if 'can_raise' in context and type(context['can_raise']) is not bool: raise ValueError('Invalid raising permission')
    seats=context.get('seats',list(range(8)))
    if not isinstance(seats,list) or any(type(x) is not int or not 0<=x<8 for x in seats): raise ValueError('Invalid seats')


def advice(snapshot, context, stale=False):
    validate_context(context)
    result={'action':None,'reason':'Waiting for a reading.','model':'Hemobot rules · not a solver','positions':{}}
    if not snapshot: return result
    fields=snapshot['fields']
    def observed(key): return fields.get(key,{}).get('value')
    hero=context.get('hero',5)
    if type(hero) is not int or not 0<=hero<8: raise ValueError('Choose a hero seat from 0 to 7.')
    seats=context.get('seats',list(range(8)))
    if not isinstance(seats,list) or any(type(x) is not int or not 0<=x<8 for x in seats): raise ValueError('Invalid seats.')
    dealer=context.get('dealer')
    if dealer is None: dealer=observed('dealer_seat')
    mapping=positions(seats,dealer)
    result['positions']=mapping
    result['hero_position']=mapping.get(hero)
    ring=observed(f'seat_{hero}.turn_ring') is True
    turn=context.get('turn','auto')
    result['turn']='Your turn' if turn=='yes' or (turn=='auto' and ring) else ('Not your turn' if turn=='no' else 'Turn not detected')
    def blocked(reason):
        result['reason']=reason
        return result
    if stale: return blocked('Reading is older than 30 seconds. Capture a fresh frame.')
    if not mapping or hero not in mapping: return blocked('Confirm the dealer and the seats dealt in, including folded players.')
    if result['turn']!='Your turn': return blocked('Waiting for your turn. A missing ring means unknown, not proof of another player’s turn.')
    hole=[observed(f'seat_{hero}.card_{i}') for i in range(2)]
    board=[observed(f'board_{i}') for i in range(5)]
    street=context.get('street','auto')
    if street=='auto': street=snapshot.get('table',{}).get('street')
    expected={'preflop':0,'flop':3,'turn':4,'river':5}.get(street)
    if expected is None: return blocked('Confirm the street; an unreadable board does not prove preflop.')
    if any(c is None for c in board[:expected]) or any(c is not None for c in board[expected:]):
        return blocked('Board readings do not match the selected street.')
    cards=hole+board[:expected]
    if any(not isinstance(c,str) or not CARD.fullmatch(c) for c in cards): return blocked('Both hero cards and the complete board must be readable.')
    if len(set(cards))!=len(cards): return blocked('Duplicate cards detected. Capture again.')
    stack=observed(f'seat_{hero}.stack')
    if stack is None or not math.isfinite(stack) or stack<=0: return blocked('A positive hero stack must be readable.')
    call=context.get('call')
    if call is None: return blocked('Confirm the amount to call (enter 0 when checking is free).')
    if type(call) not in (int,float) or not math.isfinite(call) or call<0: raise ValueError('Amount to call must be a nonnegative number.')
    raises=context.get('raises')
    if street=='preflop' and (type(raises) is not int or not 0<=raises<=20): return blocked('Confirm how many raises occurred preflop.')
    last=context.get('last_to_act')
    aggressor=context.get('aggressor')
    if last not in ('yes','no'): return blocked('Confirm whether you act last postflop among players still in the hand.')
    if street=='preflop' and raises and aggressor not in mapping: return blocked('Select the most recent preflop raiser.')
    if street!='preflop' and last=='yes' and aggressor not in mapping and aggressor!='none':
        return blocked('Confirm the preflop aggressor (or select no raise).')
    # A user acknowledgement applies only to this exact observation, not future hands.
    if context.get('confirmed_at')!=snapshot.get('timestamp'):
        return blocked('Review this frame and confirm context to show a suggestion.')
    pos={s:('BTN' if p=='BTN / SB' else p.replace('+','')) for s,p in mapping.items()}
    legal=[CALL]
    if call>0: legal.insert(0,FOLD)
    if context.get('can_raise') is True and stack>call: legal.append(RAISE)
    other=next(s for s in mapping if s!=hero)
    state=GameState(hole_cards={hero:hole},community=board[:expected],street=street,
                    stacks={hero:stack},pot=observed('pot') or 0,big_blind=1,
                    positions=pos,current_player=hero,facing_bet=call>0,
                    current_bet=call,street_invested={hero:0},
                    raises_this_street=raises or 0,preflop_aggressor=aggressor if type(aggressor) is int else None,
                    in_position_player=hero if last=='yes' else other)
    action=decide_action(state,hero,legal)
    label={FOLD:'Fold',CALL:'Call' if call else 'Check',RAISE:'Raise' if call else 'Bet'}[action]
    result.update(action=label,reason=f'Hemobot suggests {label.lower()} for the confirmed {street} state. '+
                  ('Raise size is not computed.' if action==RAISE else ''),
                  call_amount=min(call,stack),street=street)
    return result
