from gymnasium.spaces import Text

from civrealm.freeciv.utils.base_action import Action, ActionList


MSG_ACTOR_ID = 'message'
MSG_ACTION_KEY = 'chat'
MSG_MAX_LEN = 243
MSG_MIN_LEN = 0
"""
Functional prefixes:
  - `.`: send this chat message to all allies;
  - `:`: send this chat message to a player, e.g. `Player Name: Hello`
  - `>`: prevent special symbols being filtered by the system.
"""
MSG_CHARSET_DIGITS = frozenset({
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  # Digits
})
MSG_CHARSET_UPPER = frozenset({
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',  # Upper english alphabet
})
MSG_CHARSET_LOWER = frozenset({
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
    'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
    'u', 'v', 'w', 'x', 'y', 'z',  # Lower english alphabet
})
MSG_CHARSET_SPACE = frozenset({
    ' ',
})
MSG_CHARSET_SYMBOL = frozenset({
    '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*',
    '+', ',', '-', '.', '/', ':', ';', '<', '=', '>',
    '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|',
    '}', '~',
})
MSG_CHARSET = (
    MSG_CHARSET_DIGITS
    .union(MSG_CHARSET_UPPER)
    .union(MSG_CHARSET_LOWER)
    .union(MSG_CHARSET_SPACE)
    .union(MSG_CHARSET_SYMBOL)
)


class PlayerActions(ActionList):
    def __init__(self, ws_client):
        super().__init__(ws_client)
        self.add_actor(MSG_ACTOR_ID)
        self.add_action(MSG_ACTOR_ID, ActChat())

    @property
    def action_space(self):
        return Text(max_length=MSG_MAX_LEN, min_length=MSG_MIN_LEN, charset=MSG_CHARSET)

    def get_action(self, actor_id, action_key):
        if actor_id == MSG_ACTOR_ID:
            return super().get_action(actor_id, MSG_ACTION_KEY)
        return super().get_action(actor_id, action_key)

    def _can_actor_act(self, actor_id):
        return True

    def trigger_single_action(self, actor_id, action_id):
        if actor_id == MSG_ACTOR_ID:
            action_id, message = MSG_ACTION_KEY, action_id
            super().trigger_single_action(actor_id, action_id, message=message)
        else:
            super().trigger_single_action(actor_id, action_id)

    def update(self, pplayer):
        pass


class ActChat(Action):
    action_key = MSG_ACTION_KEY

    def __init__(self):
        super().__init__()
        self.message = ''

    def trigger_action(self, ws_client, **kwargs):
        self.message = kwargs['message']
        ws_client.send_message(self.message)

    def is_action_valid(self):
        return True

    def _action_packet(self):
        return {
            'pid': 26,  # PACKET_CHAT_MSG_REQ
            'message': self.message,
        }
