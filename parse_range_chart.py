import os
import numpy as np
import json
from enums.GameType import GameType
from enums.Position import Position
from enums.Value import Value
from enums.Action import Action

from ranges import BASE_CHART_DIR
from ranges.RangeChart import load_range_charts
import requests

FILENAME = "test.json"
GAME_TYPE = GameType.SixPlayer
POSITION = Position.BB
ACTION = Action.RaiseSB

FOLD = "FOLD"

URL = "https://pokercoaching.com/wp-json/pokercoaching/v1/get_charts?_wpnonce=af40fce9be"
HEADERS = {
    "Content-Type": "multipart/form-data; boundary=----WebKitFormBoundaryQuYcq3JFSpwudTgt",
    "Cookie": "sbjs_migrations=1418474375998%3D1; sbjs_first_add=fd%3D2024-01-18%2018%3A43%3A12%7C%7C%7Cep%3Dhttps%3A%2F%2Fpokercoaching.com%2Frange-analyzer%2F%7C%7C%7Crf%3Dhttps%3A%2F%2Fwww.google.com%2F; sbjs_current=typ%3Dorganic%7C%7C%7Csrc%3Dgoogle%7C%7C%7Cmdm%3Dorganic%7C%7C%7Ccmp%3D%28none%29%7C%7C%7Ccnt%3D%28none%29%7C%7C%7Cid%3D%28none%29%7C%7C%7Ctrm%3D%28none%29%7C%7C%7Cmtke%3D%28none%29; sbjs_first=typ%3Dorganic%7C%7C%7Csrc%3Dgoogle%7C%7C%7Cmdm%3Dorganic%7C%7C%7Ccmp%3D%28none%29%7C%7C%7Ccnt%3D%28none%29%7C%7C%7Cid%3D%28none%29%7C%7C%7Ctrm%3D%28none%29%7C%7C%7Cmtke%3D%28none%29; _gcl_au=1.1.2115165214.1705603393; _fbp=fb.1.1705603392904.393957895; ajs_anonymous_id=2357eed9-0267-4995-af58-55e0f1e9e8c6; sbjs_current_add=fd%3D2024-01-31%2012%3A38%3A05%7C%7C%7Cep%3Dhttps%3A%2F%2Fpokercoaching.com%2Frange-analyzer%2F%7C%7C%7Crf%3Dhttps%3A%2F%2Fwww.google.com%2F; wpf_ref=%7B%22original_ref%22%3A%22https%3A%5C%2F%5C%2Fpokercoaching.com%5C%2Fcharts%5C%2Fgto%5C%2F%3Ftype%3Dcashgame%22%2C%22landing_page%22%3A%22%5C%2Fwp-json%5C%2Fpokercoaching%5C%2Fv1%5C%2Fget_charts%22%7D; mo_openid_signup_url=https%3A%2F%2Fpokercoaching.com%2Fregister%2Fref%2F%5Bref%5D%2F; wordpress_logged_in_pocket_seises=noonenoonenoone3%7C1738240834%7CgyaOHuS7l32QPhjZngZI9sEfJfcXPb12pAaERfktVa0%7C67bb0fffb829d8d06c9d4cb4bd9a68cc821c97f8a301440fba7f09fb19497eb6; ajs_user_id=583444; sbjs_udata=vst%3D4%7C%7C%7Cuip%3D%28none%29%7C%7C%7Cuag%3DMozilla%2F5.0%20%28Windows%20NT%2010.0%3B%20Win64%3B%20x64%29%20AppleWebKit%2F537.36%20%28KHTML%2C%20like%20Gecko%29%20Chrome%2F121.0.0.0%20Safari%2F537.36; sbjs_session=pgs%3D1%7C%7C%7Ccpg%3Dhttps%3A%2F%2Fpokercoaching.com%2Fcharts%2Fgto%2F%3Ftype%3Dcash6max; mc_landing_site=https%3A%2F%2Fpokercoaching.com%2Fcharts%2Fgto%2F%3Ftype%3Dcash6max; __kla_id=eyJjaWQiOiJaalkwTXpRNE1USXRZMlExTmkwME5qZG1MVGs1TWpVdFpERXdaakprWmpjek9UUmsiLCIkcmVmZXJyZXIiOnsidHMiOjE3MDU2MDMzOTMsInZhbHVlIjoiaHR0cHM6Ly93d3cuZ29vZ2xlLmNvbS8iLCJmaXJzdF9wYWdlIjoiaHR0cHM6Ly9wb2tlcmNvYWNoaW5nLmNvbS9yYW5nZS1hbmFseXplci8ifSwiJGxhc3RfcmVmZXJyZXIiOnsidHMiOjE3MDc0Nzk5MDgsInZhbHVlIjoiIiwiZmlyc3RfcGFnZSI6Imh0dHBzOi8vcG9rZXJjb2FjaGluZy5jb20vY2hhcnRzL2d0by8/dHlwZT1jYXNoNm1heCJ9LCIkZXhjaGFuZ2VfaWQiOiJrUGJRQlFjTHpDajNRSFBzMUlmZC1SNFptLXQyWEhSVy1sU1o3Z0ZmdUdKUWloWk55SUdWVnBrOWpPVENLU3Q5LlVUZ2hhNSJ9; _ga=GA1.2.1791119619.1705603393; _gid=GA1.2.410531821.1707479908; _dc_gtm_UA-31155271-1=1; _gat=1; amplitude_idundefinedpokercoaching.com=eyJvcHRPdXQiOmZhbHNlLCJzZXNzaW9uSWQiOm51bGwsImxhc3RFdmVudFRpbWUiOm51bGwsImV2ZW50SWQiOjAsImlkZW50aWZ5SWQiOjAsInNlcXVlbmNlTnVtYmVyIjowfQ==; scroll=null; gtm_p6_ip=185.237.63.23; gtm_p6_country_code=gb; gtm_p6_country=0b407281768f0e833afef47ed464b6571d01ca4d53c12ce5c51d1462f4ad6677; gtm_p6_st=d56d0ff69b62792a00a361fbf6e02e2a634a7a8da1c3e49d59e71e0f19c27875; gtm_p6_ct=6089854c94ca5454b76be6752c562901a985f64c9a946f62976aeab593b83161; gtm_p6_zip=06f8faea3b5f697691b6d063a07ba4ffaf1ece9a1d473c588565231cdc8e59cc; gtm_p6_s_id=212338693; amplitude_id_acaae423faf755c75527a35f0b6aea02pokercoaching.com=eyJkZXZpY2VJZCI6ImZlZmQ4MDMyLTg0YmMtNDZkZS1hNDYzLTUwN2VlZjg5NjBjOFIiLCJ1c2VySWQiOiI1ODM0NDQiLCJvcHRPdXQiOmZhbHNlLCJzZXNzaW9uSWQiOjE3MDc0Nzk5MDg4MzIsImxhc3RFdmVudFRpbWUiOjE3MDc0Nzk5NTI3MTcsImV2ZW50SWQiOjI2MSwiaWRlbnRpZnlJZCI6NDQsInNlcXVlbmNlTnVtYmVyIjozMDV9; _ga_9M4Z0PND8G=GS1.1.1707479908.5.1.1707479952.16.0.0"
}


def get_chart_data():
    response = requests.post(
        url=URL,
        headers=HEADERS,
        data="""------WebKitFormBoundaryQuYcq3JFSpwudTgt
Content-Disposition: form-data; name="db"

crawler
------WebKitFormBoundaryQuYcq3JFSpwudTgt
Content-Disposition: form-data; name="rules"

true
------WebKitFormBoundaryQuYcq3JFSpwudTgt
Content-Disposition: form-data; name="action"

preflop_chart_handler
------WebKitFormBoundaryQuYcq3JFSpwudTgt
Content-Disposition: form-data; name="filter[type]"

mtts_full_ring
------WebKitFormBoundaryQuYcq3JFSpwudTgt
Content-Disposition: form-data; name="filter[blinds]"

20
------WebKitFormBoundaryQuYcq3JFSpwudTgt
Content-Disposition: form-data; name="filter[position]"

UTG+2
------WebKitFormBoundaryQuYcq3JFSpwudTgt
Content-Disposition: form-data; name="filter[action]"

RFI
------WebKitFormBoundaryQuYcq3JFSpwudTgt
Content-Disposition: form-data; name="data_format"

html
------WebKitFormBoundaryQuYcq3JFSpwudTgt--"""
    )

    data = json.loads(response.text)
    return data['chart_data']


def get_all_actions(data):
    all_actions = []
    for hand in data:
        if 'actions' in data[hand]:
            actions = data[hand]["actions"].keys()
            for action in actions:
                if action not in all_actions:
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
        action_idx = actions.index(action)
        hand_actions[action_idx] = float(value['val'])

    return hand_actions


def parse(json_str):
    data = json.loads(json_str)
    data = data["data_v2"]["data"]

    actions = get_all_actions(data)
    chart = np.zeros((15, 15, 2, len(actions)))

    for hand, value in data.items():
        idx1, idx2, idx3 = parse_hand(hand)
        hand_actions = get_action_values(value, actions)
        chart[idx1, idx2, idx3] = hand_actions

    return chart, actions


if __name__ == "__main__":
    # charts = load_range_charts()
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
    positions = []
    actions = []
