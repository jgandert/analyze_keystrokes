import random

TIME_I = 0
IS_DOWN_I = 1
KEY_I = 2

NON_MOD_VALUE = False
MOD_VALUE = True

# Modifier keys
MOD_KEYS = {'alt', 'alt_gr', 'alt_l', 'alt_r', 'cmd', 'cmd_r', 'ctrl', 'ctrl_l', 'ctrl_r', 'shift',
            'shift_r'}

# yes, this isn't perfect
LEFT_SIDE_KEYS = [
    "q", "w", "e", "r", "t",
    "a", "s", "d", "f", "g",
    "z", "x", "c", "v", "b",
    'shift_l', 'ctrl_l', 'alt_l'
]

RIGHT_SIDE_KEYS = [
    "y", "u", "i", "o", "p",
    "h", "j", "k", "l",
    "n", "m", 'backspace', 'enter', 'delete'
                                    'shift_r', 'ctrl_r', 'alt_r'
]


def is_left_not_right(key):
    if key in RIGHT_SIDE_KEYS:
        return False
    if key in LEFT_SIDE_KEYS:
        return True
    return bool(random.getrandbits(1))


MAIN_AREA_KEYS = set("abcdefghijklmnopqrstuvwxyzäöüß,.<")

# We disabled the following for one of two reasons
# 1. on a traditional keyboard they are far away (e.g. left key), possibly skewing the timing, or
# 2. they are most likely not an actual shortcut (ctrl + left is, but shift + backspace is not)
# MAIN_AREA_KEYS.update('left right backspace space enter tab'.split())


VK_CODE_TO_LETTER = {
    '8': 'backspace', '9': 'tab', '12': 'clear', '13': 'enter', '16': 'SHIFT', '17': 'ctrl',
    '18': 'alt', '19': 'pause', '20': 'caps_lock', '27': 'esc', '32': 'space', '33': 'page_up',
    '34': 'page_down', '35': 'end', '36': 'home', '37': 'left', '38': 'up', '39': 'right',
    '40': 'down', '41': 'select', '42': 'print', '43': 'execute', '44': 'print_screen',
    '45': 'insert', '46': 'delete', '47': 'help', '48': '0', '49': '1', '50': '2', '51': '3',
    '52': '4', '53': '5', '54': '6', '55': '7', '56': '8', '57': '9', '65': 'a', '66': 'b',
    '67': 'c', '68': 'd', '69': 'e', '70': 'f', '71': 'g', '72': 'h', '73': 'i', '74': 'j',
    '75': 'k', '76': 'l', '77': 'm', '78': 'n', '79': 'o', '80': 'p', '81': 'q', '82': 'r',
    '83': 's', '84': 't', '85': 'u', '86': 'v', '87': 'w', '88': 'x', '89': 'y', '90': 'z',
    '91': 'cmd_l', '92': 'cmd_r', '93': 'menu', '95': 'sleep', '96': 'num_0', '97': 'num_1',
    '98': 'num_2', '99': 'num_3', '100': 'num_4', '101': 'num_5', '102': 'num_6', '103': 'num_7',
    '104': 'num_8', '105': 'num_9', '106': '*', '107': '+', '108': 'separator', '109': '-',
    '110': '.', '111': '/', '112': 'f1', '113': 'f2', '114': 'f3', '115': 'f4', '116': 'f5',
    '117': 'f6', '118': 'f7', '119': 'f8', '120': 'f9', '121': 'f10', '122': 'f11', '123': 'f12',
    '124': 'f13', '125': 'f14', '126': 'f15', '127': 'f16', '128': 'f17', '129': 'f18',
    '130': 'f19', '131': 'f20', '132': 'f21', '133': 'f22', '134': 'f23', '135': 'f24',
    '144': 'num_lock', '145': 'scroll_lock', '160': 'shift_l', '161': 'shift_r', '162': 'ctrl_l',
    '163': 'ctrl_r', '164': 'alt_l', '165': 'alt_r', '166': 'browser_back',
    '167': 'browser_forward', '168': 'browser_refresh', '169': 'browser_stop',
    '170': 'browser_search', '171': 'browser_favorites', '172': 'browser_start_and_home',
    '173': 'media_volume_mute', '174': 'media_volume_down', '175': 'media_volume_up',
    '176': 'media_next', '177': 'media_previous', '178': 'media_stop', '179': 'media_play_pause',
    '180': 'start_mail', '181': 'select_media', '182': 'start_application_1',
    '183': 'start_application_2', '186': ';', '187': '+', '188': ',', '189': '-', '190': '.',
    '191': '/', '219': '[', '220': '\\', '221': ']', '222': "'", '226': '<'
}

LAYOUT_TO_SHIFTED_KEY_TO_KEY = {
    "qwerty": {"!": "1", "@": "2", "#": "3", "$": "4", "%": "5", "^": "6", "&": "7", "*": "8",
               "(": "9", ")": "0", "?": "/", "{": "[", "}": "]", "_": "-", ":": ";", "<": ",",
               ">": ".", '"': "'", "|": "\\", "+": "="},
    "qwertz": {"!": "1", "\"": "2", "§": "3", "$": "4", "%": "5", "&": "6", "/": "7", "(": "8",
               ")": "9", "=": "0", "?": "ß", "*": "+", "'": "#", "_": "-", ":": ".", ";": ",",
               ">": "<", "~": "+", "°": "^"},
    "azerty": {"&": "1", "é": "2", '"': "3", "'": "4", "(": "5", "-": "6", "è": "7", "_": "8",
               "ç": "9", "à": "0", ")": "°", "+": "=", "?": ",", ".": ";", "/": ":", "§": "!"}
}

UPPERCASE = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
             'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'À', 'Á', 'Ä', 'Å', 'Ç', 'È', 'É', 'Í',
             'Ï', 'Ñ', 'Ô', 'Ö', 'Ø', 'Ù', 'Ü'}

LAYOUT_TO_UPPERCASE = {
    layout: UPPERCASE | values.keys() for (layout, values) in LAYOUT_TO_SHIFTED_KEY_TO_KEY.items()
}

SUPPORTED_LAYOUTS = LAYOUT_TO_UPPERCASE.keys()
