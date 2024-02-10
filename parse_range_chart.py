import os
import numpy as np
import json
from enums.Card import Card
from enums.GameType import GameType
from enums.OpponentAction import OpponentAction
from enums.Position import GAME_TYPE_POSITIONS, Position
from enums.Suit import Suit
from enums.Value import Value
from enums.Action import Action
from poker.GameState import GameState
from poker.PreFlopGameState import PreFlopGameState

from ranges import BASE_CHART_DIR
from ranges.RangeChart import load_range_charts
import requests

FILENAME = "test.json"
GAME_TYPE = GameType.SixPlayer
POSITION = Position.BB
ACTION = Action.RaiseSB

FOLD = "FOLD"
NON_ACTIONS = ['oor']
URL = "https://pokercoaching.com/wp-json/pokercoaching/v1/get_charts?_wpnonce=9e01ce7db4"
HEADERS = {
    "Content-Type": "multipart/form-data; boundary=----WebKitFormBoundaryA5PjghKfYmgi0CWq",
    "Cookie": "sbjs_migrations=1418474375998%3D1; sbjs_first_add=fd%3D2024-01-18%2018%3A43%3A12%7C%7C%7Cep%3Dhttps%3A%2F%2Fpokercoaching.com%2Frange-analyzer%2F%7C%7C%7Crf%3Dhttps%3A%2F%2Fwww.google.com%2F; sbjs_current=typ%3Dorganic%7C%7C%7Csrc%3Dgoogle%7C%7C%7Cmdm%3Dorganic%7C%7C%7Ccmp%3D%28none%29%7C%7C%7Ccnt%3D%28none%29%7C%7C%7Cid%3D%28none%29%7C%7C%7Ctrm%3D%28none%29%7C%7C%7Cmtke%3D%28none%29; sbjs_first=typ%3Dorganic%7C%7C%7Csrc%3Dgoogle%7C%7C%7Cmdm%3Dorganic%7C%7C%7Ccmp%3D%28none%29%7C%7C%7Ccnt%3D%28none%29%7C%7C%7Cid%3D%28none%29%7C%7C%7Ctrm%3D%28none%29%7C%7C%7Cmtke%3D%28none%29; _gcl_au=1.1.2115165214.1705603393; _fbp=fb.1.1705603392904.393957895; ajs_anonymous_id=2357eed9-0267-4995-af58-55e0f1e9e8c6; sbjs_current_add=fd%3D2024-01-31%2012%3A38%3A05%7C%7C%7Cep%3Dhttps%3A%2F%2Fpokercoaching.com%2Frange-analyzer%2F%7C%7C%7Crf%3Dhttps%3A%2F%2Fwww.google.com%2F; wpf_ref=%7B%22original_ref%22%3A%22https%3A%5C%2F%5C%2Fpokercoaching.com%5C%2Fcharts%5C%2Fgto%5C%2F%3Ftype%3Dcashgame%22%2C%22landing_page%22%3A%22%5C%2Fwp-json%5C%2Fpokercoaching%5C%2Fv1%5C%2Fget_charts%22%7D; mo_openid_signup_url=https%3A%2F%2Fpokercoaching.com%2Fregister%2Fref%2F%5Bref%5D%2F; wordpress_logged_in_pocket_seises=noonenoonenoone3%7C1738240834%7CgyaOHuS7l32QPhjZngZI9sEfJfcXPb12pAaERfktVa0%7C67bb0fffb829d8d06c9d4cb4bd9a68cc821c97f8a301440fba7f09fb19497eb6; ajs_user_id=583444; mc_landing_site=https%3A%2F%2Fpokercoaching.com%2Fcharts%2Fgto%2F%3Ftype%3Dcash6max; _gid=GA1.2.410531821.1707479908; amplitude_idundefinedpokercoaching.com=eyJvcHRPdXQiOmZhbHNlLCJzZXNzaW9uSWQiOm51bGwsImxhc3RFdmVudFRpbWUiOm51bGwsImV2ZW50SWQiOjAsImlkZW50aWZ5SWQiOjAsInNlcXVlbmNlTnVtYmVyIjowfQ==; gtm_p6_ip=185.237.63.23; gtm_p6_country_code=gb; gtm_p6_country=0b407281768f0e833afef47ed464b6571d01ca4d53c12ce5c51d1462f4ad6677; gtm_p6_st=d56d0ff69b62792a00a361fbf6e02e2a634a7a8da1c3e49d59e71e0f19c27875; gtm_p6_ct=6089854c94ca5454b76be6752c562901a985f64c9a946f62976aeab593b83161; gtm_p6_zip=06f8faea3b5f697691b6d063a07ba4ffaf1ece9a1d473c588565231cdc8e59cc; gtm_p6_s_id=212338693; gtm_p6_g_clid=null; gtm_p6_tt_clid=null; wp_woocommerce_session_pocket_seises=583444%7C%7C1707662896%7C%7C1707659296%7C%7C9dc551f4dc2ae840faa9cf983a93f780; sbjs_udata=vst%3D6%7C%7C%7Cuip%3D%28none%29%7C%7C%7Cuag%3DMozilla%2F5.0%20%28Windows%20NT%2010.0%3B%20Win64%3B%20x64%29%20AppleWebKit%2F537.36%20%28KHTML%2C%20like%20Gecko%29%20Chrome%2F121.0.0.0%20Safari%2F537.36; scroll=null; __stripe_mid=23c4020f-da87-4c43-9a50-3af8dce665c5ff8097; __stripe_sid=a288270f-d251-45dc-9dd7-2a4c4a78992003255c; _gat=1; _ga=GA1.2.1791119619.1705603393; sbjs_session=pgs%3D9%7C%7C%7Ccpg%3Dhttps%3A%2F%2Fpokercoaching.com%2Fcharts%2Fgto%2F%3Ftype%3Dmttsfullring; __kla_id=eyJjaWQiOiJORE5oTXpRNVpqSXROemxrWWkwME1EUTFMVGcyTkRZdE9XVTBZVGRsTTJGaVlqUTEiLCIkcmVmZXJyZXIiOnsidHMiOjE3MDc0OTAxNTYsInZhbHVlIjoiIiwiZmlyc3RfcGFnZSI6Imh0dHBzOi8vcG9rZXJjb2FjaGluZy5jb20vY2FydC8/dHlwZT1tdHRzZnVsbHJpbmcmYWRkLXRvLWNhcnQ9MjA0Mzk1OSJ9LCIkbGFzdF9yZWZlcnJlciI6eyJ0cyI6MTcwNzQ5MDI0NiwidmFsdWUiOiIiLCJmaXJzdF9wYWdlIjoiaHR0cHM6Ly9wb2tlcmNvYWNoaW5nLmNvbS9jYXJ0Lz90eXBlPW10dHNmdWxscmluZyZhZGQtdG8tY2FydD0yMDQzOTU5In0sIiRleGNoYW5nZV9pZCI6ImtQYlFCUWNMekNqM1FIUHMxSWZkLVI0Wm0tdDJYSFJXLWxTWjdnRmZ1R0pRaWhaTnlJR1ZWcGs5ak9UQ0tTdDkuVVRnaGE1In0=; amplitude_id_acaae423faf755c75527a35f0b6aea02pokercoaching.com=eyJkZXZpY2VJZCI6ImZlZmQ4MDMyLTg0YmMtNDZkZS1hNDYzLTUwN2VlZjg5NjBjOFIiLCJ1c2VySWQiOiI1ODM0NDQiLCJvcHRPdXQiOmZhbHNlLCJzZXNzaW9uSWQiOjE3MDc0ODc1MjQxMjYsImxhc3RFdmVudFRpbWUiOjE3MDc0OTAyNjAxMjUsImV2ZW50SWQiOjM4NiwiaWRlbnRpZnlJZCI6NjQsInNlcXVlbmNlTnVtYmVyIjo0NTB9; _ga_9M4Z0PND8G=GS1.1.1707487524.7.1.1707490260.20.0.0"
}

position_strings = {
    Position.UTG: 'UTG',
    Position.UTG1: 'UTG+1',
    Position.LJ: 'UTG+2',
    Position.HJ: 'UTG+3',
    Position.CO: 'CO',
    Position.BTN: "BTN",
    Position.SB: "SB",
    Position.BB: "BB",
}

action_strings = {
    OpponentAction.RFI: "RFI",
    OpponentAction.RAISE: "VS RAISE",
    OpponentAction.ALL_IN: "VS ALLIN",
    OpponentAction.THREE_BET: "VS 3BET",
    OpponentAction.THREE_BET_ALL_IN: "VS 3BET ALLIN",
    OpponentAction.FOUR_BET_ALL_IN: "VS 4BET ALLIN",
}

""""""


def get_chart_data(blinds, position, action, opposition):
    opposition_data = f"""------WebKitFormBoundaryA5PjghKfYmgi0CWq
Content-Disposition: form-data; name="filter[opposition]"

{opposition}
""" if opposition is not None else ''
    data = f"""------WebKitFormBoundaryA5PjghKfYmgi0CWq
Content-Disposition: form-data; name="db"

crawler
------WebKitFormBoundaryA5PjghKfYmgi0CWq
Content-Disposition: form-data; name="rules"

true
------WebKitFormBoundaryA5PjghKfYmgi0CWq
Content-Disposition: form-data; name="action"

preflop_chart_handler
------WebKitFormBoundaryA5PjghKfYmgi0CWq
Content-Disposition: form-data; name="filter[type]"

mtts_full_ring
------WebKitFormBoundaryA5PjghKfYmgi0CWq
Content-Disposition: form-data; name="filter[blinds]"

{blinds}
------WebKitFormBoundaryA5PjghKfYmgi0CWq
Content-Disposition: form-data; name="filter[position]"

{position}
------WebKitFormBoundaryA5PjghKfYmgi0CWq
Content-Disposition: form-data; name="filter[action]"

{action}
------WebKitFormBoundaryA5PjghKfYmgi0CWq
Content-Disposition: form-data; name="data_format"

html
{opposition_data}------WebKitFormBoundaryA5PjghKfYmgi0CWq--"""
    response = requests.post(
        url=URL,
        headers=HEADERS,
        data=data
    )

    return json.loads(response.text)


def get_all_actions(data):
    all_actions = []
    for hand in data:
        if 'actions' in data[hand]:
            actions = data[hand]["actions"].keys()
            for action in actions:
                if action not in all_actions and action not in NON_ACTIONS:
                    all_actions.append(action)

    if FOLD not in all_actions:
        all_actions.append(FOLD)

    return all_actions


def parse_hand(hand):
    value_1 = Value.from_string(hand[0])
    value_2 = Value.from_string(hand[1])

    if len(hand) == 2:
        return value_1.value, value_2.value, 1

    return value_1.value, value_2.value, int(hand[2] == "s")


def get_action_values(hand, actions):
    hand_actions = np.zeros(len(actions))
    fold_idx = actions.index(FOLD)

    if "actions" not in hand:
        hand_actions[fold_idx] = 1.0
        return hand_actions

    for action, value in hand["actions"].items():
        if action in NON_ACTIONS:
            continue
        action_idx = actions.index(action)
        hand_actions[action_idx] = float(value['val'])

    return hand_actions


def parse(data):
    data = json.loads(data["data_v2"])
    data = data["data"]

    actions = get_all_actions(data)
    chart = np.zeros((15, 15, 2, len(actions)))

    for hand, value in data.items():
        idx1, idx2, idx3 = parse_hand(hand)
        hand_actions = get_action_values(value, actions)
        chart[idx1, idx2, idx3] = hand_actions

    return chart, actions


def get_and_save_data(game_type, blinds, position, action, opposition):
    data = get_chart_data(
        blinds, position_strings[position], action_strings[action], position_strings[opposition] if opposition is not None else None)
    if "chart_data" not in data:
        raise Exception("chart data not found!")

    charts, actions = parse(data["chart_data"])

    base_filename = f"{game_type.value}-{blinds}-{position.value}-{action.value}-{opposition.value if opposition is not None else 'none'}"

    filename = f"{base_filename}-chart.npy"
    save_path = os.path.join(BASE_CHART_DIR, filename)
    np.save(save_path, charts)

    with open(os.path.join(BASE_CHART_DIR, f"{base_filename}-actions.txt"), 'w') as f:
        f.write(",".join(actions))


if __name__ == "__main__":
    # charts = load_range_charts()
    # g = PreFlopGameState(GameType.NinePlayer, GameState(GameType.NinePlayer, 2, 5, 0, []), [Card(
    #     Suit(1), Value(1)), Card(Suit(2), Value(1))], charts[GameType.NinePlayer.value][Position.BTN.value])
    # print(str(g))
    # ujgyk
    # f = 0
    # chart = charts[GAME_TYPE][POSITION][ACTION]
    # hand = Hand(Card(Suit(3), Value(11)), Card(Suit(3), Value(10)))
    # print(chart[hand])
    # with open(FILENAME, 'r') as f:
    #     chart, actions = parse(f.read())
    #     base_filename = f"{GAME_TYPE.value}-{POSITION.value}-{ACTION.value}"

    #     filename = f"{base_filename}-chart.npy"
    #     save_path = os.path.join(BASE_CHART_DIR, filename)
    #     np.save(save_path, chart)

    #     with open(os.path.join(BASE_CHART_DIR, f"{base_filename}-actions.txt"), 'w') as f:
    #         f.write(",".join(actions))

    game_type = GameType.NinePlayer
    blinds = 10
    positions = GAME_TYPE_POSITIONS[game_type]
    actions = [action for action in OpponentAction]

    successes = 0
    failures = 0
    errors = []

    for position in positions:

        # no ranges for utg2
        if position == Position.UTG2:
            continue

        for action in actions:
            print(blinds, position.value, action.value)

            # RFI has no opponent
            if action == OpponentAction.RFI:
                try:

                    get_and_save_data(game_type, blinds,
                                      position, action, None)
                    successes += 1
                except Exception as e:
                    failures += 1
                    errors.append(f"{position}-{action}-none - {str(e)}")
            else:
                for opponent in positions:
                    if opponent == position:
                        continue
                    try:
                        get_and_save_data(game_type, blinds,
                                          position, action, opponent)
                        successes += 1
                    except Exception as e:
                        failures += 1
                        print(e)
                        errors.append(f"{position}-{action}-none - {str(e)}")
    print(f"{successes} sucess, {failures} failures")
    for error in errors:
        print(error)
