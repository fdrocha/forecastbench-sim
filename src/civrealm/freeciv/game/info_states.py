# Copyright (C) 2023  The CivRealm project
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.

from civrealm.freeciv.utils.base_state import PlainState, DictState

WELCOME_TEMPLATE = (
    'Welcome, {username} ruler of the {nation} empire.'
    'Your task is to create a great empire! You should start by '
    'exploring the land around you with your explorer, '
    'and using your settlers to find a good place to build '
    'a city. Right-click with the mouse on your units for a list of available '
    'orders such as move, explore, build cities and attack. '
    'Good luck, and have a lot of fun!'
)


class GameState(PlainState):
    def __init__(self, player_ctrl, scenario_info, calendar_info):
        super().__init__()
        self.player_ctrl = player_ctrl
        self.scenario_info = scenario_info
        self.calendar_info = calendar_info
        self.chat_messages = ''
        self._update_message = self._update_message_welcome

    def _update_state(self, pplayer):
        if pplayer != None:
            self._state.update(self.calendar_info)
            del self._state['calendar_fragment_name']
            self._state.update(self.scenario_info)
            self._update_message()

    def _update_message_welcome(self):
        my_player = self.player_ctrl.my_player
        self.chat_messages = WELCOME_TEMPLATE.format(
            username=my_player['name'],
            nation=my_player['nation'],
        )
        self._update_message = self._update_message_play
        self._update_message_play()

    def _update_message_play(self):
        self._state['messages'] = self.chat_messages
        self.chat_messages = ''


class ServerState(DictState):
    def __init__(self, server_settings):
        super().__init__()
        self.server_settings = server_settings

    def _update_state(self, pplayer):
        if pplayer != None:
            pass
            # TODO: Treating fields that are lists to be treated later on
            # self._state.update(self.server_settings)


class RuleState(PlainState):
    def __init__(self, game_info):
        super().__init__()
        self.game_info = game_info

    def _update_state(self, pplayer):
        if pplayer != None:
            self._state.update(self.game_info)
            # TODO: Treating fields that are lists to be treated later on
            for key in ["global_advances", "granary_food_ini", "great_wonder_owners",
                        "min_city_center_output", "diplchance_initial_odds"]:
                del self._state[key]
