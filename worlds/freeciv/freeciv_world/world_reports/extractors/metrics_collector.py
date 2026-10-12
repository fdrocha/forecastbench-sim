"""Metrics collection from game states and savegames

This module extracts all world report data from state files and savegames,
organizing it into a flat, metric-based structure.
"""

import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict

from ..utils import metrics
from ..utils.event_detector import EventDetector
from ..utils.savegame_parser import (
    get_savegame_data_for_report,
    reset_savegame_coverage,
    compute_savegame_coverage,
    count_wonders_from_savegame,
    military_unit_count,
)

# Marker stamped into metadata["truth_source"] when the per-player metrics were
# overridden from the omniscient server savegame (vs the fog-of-war client view)
TRUTH_SOURCE_SAVEGAME = "savegame"
SAVEGAME_TRUTH_SERIES = [
    "scores", "treasury", "cities_count", "population", "territory_size",
    "units_count", "military_units_count", "wonders_count", "is_alive", "government",
]

# Diplomatic state constants (from player_const.py)
DS_ARMISTICE = 0
DS_WAR = 1
DS_CEASEFIRE = 2
DS_PEACE = 3
DS_ALLIANCE = 4
DS_NO_CONTACT = 5
DS_TEAM = 6

DS_TXT = {
    DS_ARMISTICE: "Armistice",
    DS_WAR: "War",
    DS_CEASEFIRE: "Ceasefire",
    DS_PEACE: "Peace",
    DS_ALLIANCE: "Alliance",
    DS_NO_CONTACT: "No contact",
    DS_TEAM: "Team",
}


def select_snapshot_turns(max_turn: int, max_snapshots: int = 5,
                         min_turn: int = 5, min_spacing: int = 10) -> List[int]:
    """Select snapshot turns for territorial maps using backward selection

    Args:
        max_turn: The latest turn in the report
        max_snapshots: Maximum number of snapshots to include (default: 5)
        min_turn: Earliest turn to consider (default: 5)
        min_spacing: Minimum turns between snapshots (default: 10)

    Returns:
        Sorted list of turn numbers for snapshot generation
    """
    if max_turn < min_turn:
        return []

    if max_turn == min_turn:
        return [min_turn]

    available_range = max_turn - min_turn
    ideal_spacing = available_range // (max_snapshots - 1) if max_snapshots > 1 else available_range
    spacing = max(min_spacing, ideal_spacing)

    snapshots = [max_turn]
    current = max_turn

    while len(snapshots) < max_snapshots:
        next_turn = current - spacing
        if next_turn < min_turn:
            if min_turn not in snapshots and (current - min_turn) >= min_spacing:
                snapshots.append(min_turn)
            break
        snapshots.append(next_turn)
        current = next_turn

    return sorted(snapshots)


class MetricsCollector:
    """Collects and organizes all world report metrics from game data"""

    def __init__(self):
        """Initialize metrics collector"""
        # Provenance of per-player metrics; filled by collect_time_series and
        # stamped into metadata["truth_source"] by collect_all.
        self.truth_source: Dict[str, Any] = {
            "source": None,
            "turns_total": 0,
            "turns_overridden": 0,
            "turns_without_savegame": [],
            "cities_crosscheck_mismatches": [],
        }

    def collect_all(
        self,
        states: Dict[int, Dict],
        config: Any,
        data_loader: Any
    ) -> Dict[str, Any]:
        """Collect all metrics and return complete data structure

        Args:
            states: Dict mapping turn numbers to game states
            config: ReportConfig instance
            data_loader: DataLoader instance

        Returns:
            Complete world report data dictionary
        """
        if not states:
            raise ValueError("No states provided")

        sorted_turns = sorted(states.keys())
        max_turn = max(sorted_turns)

        # Start a fresh savegame-coverage log for this serialization
        recording_dir = getattr(config, 'recording_dir', None)
        if recording_dir:
            reset_savegame_coverage(recording_dir)

        print("  Collecting metadata...")
        metadata = self.collect_metadata(states, config)

        print("  Collecting civilizations...")
        civilizations = self.collect_civilizations(states, data_loader, config)

        # Update civilization count in metadata to match what we're actually displaying
        metadata["num_civilizations"] = len(civilizations)

        print("  Detecting events...")
        events = self.collect_events(states, data_loader, config)

        print("  Collecting time series data...")
        # Pass events so we can count techs from tech_discovered events
        time_series = self.collect_time_series(states, config, civilizations, data_loader, events)

        print("  Collecting snapshots...")
        # Collect snapshots at multiple turns, not just max_turn
        snapshot_turns = select_snapshot_turns(max_turn, max_snapshots=5)
        snapshots = self.collect_snapshots(states, snapshot_turns, config)

        print("  Collecting territory snapshots...")
        # Actually capture territory data at selected turns
        territory_snapshot_turns = select_snapshot_turns(max_turn, max_snapshots=3)
        territory_snapshots = self.collect_territory_snapshots(states, territory_snapshot_turns)

        print("  Collecting diplomacy data...")
        # Use savegame-based diplomacy extraction (has all pairwise relationships)
        # Note: collect_diplomacy_from_states is buggy - state['dipl'] only contains
        # the controlled player's relationships, not all pairwise diplomatic states.
        diplomacy = self.collect_diplomacy(states, config, civilizations)

        # Stamp savegame coverage into the serialization metadata. A turn
        # counts as covered only if its savegame was found AND parsed; missing
        # or failed turns silently degrade extraction to the embassy-gated
        # player view, which is exactly the dark-world failure mode.
        # Provenance of the per-player metrics: "savegame" means cities/units/
        # population/territory/treasury/scores were overridden from the
        # omniscient server savegame for every turn that had one (see
        # collect_time_series). Downstream gates key off this marker.
        metadata["truth_source"] = dict(self.truth_source)

        if recording_dir:
            cov = compute_savegame_coverage(recording_dir, sorted_turns)
            metadata["savegame_coverage"] = cov["coverage"]
            metadata["savegame_turns_expected"] = cov["expected"]
            metadata["savegame_turns_parsed"] = cov["parsed"]
            metadata["savegame_turns_missing"] = cov["missing_turns"]
            metadata["savegame_turns_failed"] = cov["failed_turns"]
            if cov["coverage"] < 1.0:
                msg = (
                    f"savegame_coverage={cov['coverage']:.4f} "
                    f"({cov['parsed']}/{cov['expected']} turns parsed; "
                    f"missing={cov['missing_turns'][:10]}"
                    f"{'...' if len(cov['missing_turns']) > 10 else ''}, "
                    f"failed={cov['failed_turns'][:10]}"
                    f"{'...' if len(cov['failed_turns']) > 10 else ''}) "
                    f"for recording_dir={recording_dir}")
                print(f"  WARNING: incomplete savegame coverage: {msg}")
                if os.environ.get("FBSIM_REQUIRE_SAVEGAME_COVERAGE") == "1":
                    raise RuntimeError(
                        f"FBSIM_REQUIRE_SAVEGAME_COVERAGE=1 and {msg}")

        return {
            "metadata": metadata,
            "civilizations": civilizations,
            "time_series": time_series,
            "events": events,
            "snapshots": snapshots,
            "territory_snapshots": territory_snapshots,
            "diplomacy": diplomacy
        }

    def collect_metadata(self, states: Dict[int, Dict], config: Any) -> Dict[str, Any]:
        """Collect report metadata

        Args:
            states: Dict mapping turn numbers to game states
            config: ReportConfig instance

        Returns:
            Metadata dictionary
        """
        sorted_turns = sorted(states.keys())
        max_turn = max(sorted_turns)
        last_state = states[max_turn]

        # Extract username from recording_dir if available
        username = "unknown"
        if hasattr(config, 'recording_dir'):
            # Recording dir is typically: logs/recordings/username/
            parts = config.recording_dir.rstrip('/').split('/')
            if len(parts) >= 2:
                username = parts[-1]

        # Map size
        map_state = last_state.get('map', {})
        map_size = [
            map_state.get('xsize', 0),
            map_state.get('ysize', 0)
        ]

        # Count civilizations
        players = last_state.get('player', {})
        num_civs = len([p for p in players.values() if isinstance(p, dict)])

        return {
            "turn": max_turn,
            "turns_analyzed": sorted_turns,
            "generated_at": datetime.now().isoformat(),
            "username": username,
            "map_size": map_size,
            "num_civilizations": num_civs
        }

    def collect_civilizations(
        self,
        states: Dict[int, Dict],
        data_loader: Any,
        config: Any
    ) -> Dict[int, Dict[str, Any]]:
        """Collect civilization information

        Args:
            states: Dict mapping turn numbers to game states
            data_loader: DataLoader instance
            config: ReportConfig instance

        Returns:
            Dict mapping player_id to civilization info
        """
        civilizations = {}

        # Collect all player IDs
        all_player_ids = set()
        for state in states.values():
            if 'player' in state:
                for pid in state['player'].keys():
                    all_player_ids.add(int(pid))

        # Get names and nation info for each player
        for player_id in all_player_ids:
            # Find the latest state where this player exists
            for turn in sorted(states.keys(), reverse=True):
                state = states[turn]
                if 'player' not in state:
                    continue

                pid_str = str(player_id)
                player_info = state['player'].get(pid_str) or state['player'].get(player_id)

                if isinstance(player_info, dict):
                    nation_id = player_info.get('nation')
                    civ_name = self._get_nation_name(nation_id, data_loader)

                    # If no civilization name, try savegame data
                    if not civ_name:
                        sorted_turns = sorted(states.keys())
                        for turn in reversed(sorted_turns):
                            savegame_data = get_savegame_data_for_report(config, turn)
                            if savegame_data and 'nations' in savegame_data:
                                if player_id in savegame_data['nations']:
                                    nation_identifier = savegame_data['nations'][player_id]
                                    civ_name = self._get_nation_name_from_savegame(
                                        nation_identifier, data_loader
                                    )
                                    if civ_name:
                                        break

                    # Fallback to player name
                    if not civ_name:
                        civ_name = player_info.get('name', f'Player {player_id}')

                    civilizations[player_id] = {
                        "name": civ_name,
                        "adjective": civ_name,  # Same as name for now
                        "nation_id": nation_id if nation_id is not None else 0
                    }
                    break

        # Include all players that appear in the game (including barbarians, pirates, etc.)
        # Don't filter by is_alive so we can see all factions in charts
        return civilizations

    def collect_time_series(
        self,
        states: Dict[int, Dict],
        config: Any,
        civilizations: Dict[int, Dict],
        data_loader: Any = None,
        events: List[Dict] = None,
    ) -> Dict[str, Dict[int, Dict[int, float]]]:
        """Collect all time series metrics

        Args:
            states: Dict mapping turn numbers to game states
            config: ReportConfig instance
            civilizations: Dict of civilization info
            data_loader: DataLoader instance
            events: List of event dicts (for counting techs from discoveries)

        Returns:
            Dict mapping metric_name to {turn: {player_id: value}}
        """
        sorted_turns = sorted(states.keys())
        player_ids = list(civilizations.keys())
        events = events or []

        # Build cumulative tech counts from tech_discovered events
        # This is more reliable than reading from state files (which have None values)
        tech_discoveries_by_player = defaultdict(list)
        for event in events:
            if event.get("type") == "tech_discovered":
                pid = event.get("player_id")
                turn = event.get("turn", 0)
                if pid is not None:
                    tech_discoveries_by_player[pid].append(turn)

        # Sort discoveries by turn for cumulative counting
        for pid in tech_discoveries_by_player:
            tech_discoveries_by_player[pid].sort()

        # Initialize data structures
        time_series = {
            "treasury": {},
            "population": {},
            "science": {},
            "scores": {},  # NEW: track scores over time
            "territory_size": {},
            "arable_land": {},
            "food_production": {},
            "shield_production": {},
            "trade_production": {},
            "culture": {},
            "techs_known": {},
            "cities_count": {},
            "units_count": {},
            "military_units_count": {},
            "wonders_count": {},
            "is_alive": {},     # bool per player per turn (savegame is_alive)
            "government": {},   # government name per player per turn
        }

        # Provenance bookkeeping for the savegame override (stamped into
        # metadata["truth_source"] by collect_all).
        ruleset = data_loader.ruleset if data_loader is not None else None
        self.truth_source = {
            "source": TRUTH_SOURCE_SAVEGAME,
            "series": SAVEGAME_TRUTH_SERIES,
            "military_units_source": (
                "savegame unit table, types with attack > 0 (classic/civ2civ3 "
                "table; matches the client type_attack_strength > 0 metric); "
                "[scoreN].units for unknown rulesets"),
            "wonders_source": "client",  # flips to savegame+ruleset when possible
            "turns_total": len(sorted_turns),
            "turns_overridden": 0,
            "turns_without_savegame": [],
            "cities_crosscheck_mismatches": [],
        }

        # Collect data for each turn
        for turn in sorted_turns:
            state = states[turn]

            # Initialize turn data
            for metric in time_series:
                time_series[metric][turn] = {}

            if 'player' not in state:
                continue

            # Extract metrics for each player
            for pid_str in state['player'].keys():
                pid = int(pid_str)
                if pid not in player_ids:
                    continue

                player_info = state['player'][pid_str]
                if not isinstance(player_info, dict):
                    continue

                # Basic player metrics (client view; overridden from the
                # savegame below whenever one exists for this turn)
                time_series["is_alive"][turn][pid] = bool(player_info.get('is_alive', True))
                time_series["government"][turn][pid] = player_info.get('government_name') or ''
                time_series["treasury"][turn][pid] = metrics.get_player_gold(state, pid)
                time_series["science"][turn][pid] = metrics.get_player_science_production(state, pid)
                time_series["culture"][turn][pid] = player_info.get('culture', 0)
                time_series["scores"][turn][pid] = player_info.get('score', 0)

                # Count techs from events (cumulative count of discoveries up to this turn)
                tech_count = sum(1 for t in tech_discoveries_by_player.get(pid, []) if t <= turn)
                time_series["techs_known"][turn][pid] = tech_count

                # Aggregated city metrics
                time_series["population"][turn][pid] = metrics.aggregate_city_metric(state, pid, 'size')
                time_series["food_production"][turn][pid] = metrics.aggregate_city_metric(state, pid, 'prod_food')
                time_series["shield_production"][turn][pid] = metrics.aggregate_city_metric(state, pid, 'prod_shield')
                time_series["trade_production"][turn][pid] = metrics.aggregate_city_metric(state, pid, 'prod_trade')

                # Territory metrics
                time_series["territory_size"][turn][pid] = metrics.calculate_territory_size(state, pid)
                time_series["arable_land"][turn][pid] = metrics.calculate_arable_land(state, pid)

                # Count cities and units
                cities = state.get('city', {})
                time_series["cities_count"][turn][pid] = len([
                    c for c in cities.values()
                    if isinstance(c, dict) and c.get('owner') == pid
                ])

                units = state.get('unit', {})
                time_series["units_count"][turn][pid] = len([
                    u for u in units.values()
                    if isinstance(u, dict) and u.get('owner') == pid
                ])

                # Count military units (units with attack_strength > 0)
                time_series["military_units_count"][turn][pid] = len([
                    u for u in units.values()
                    if isinstance(u, dict) and u.get('owner') == pid
                    and u.get('type_attack_strength', 0) > 0
                ])

                # Count wonders (from city improvements)
                time_series["wonders_count"][turn][pid] = metrics.count_player_wonders(
                    state, pid, data_loader.ruleset if data_loader else None
                )

            # Try to override with complete savegame data
            savegame_data = get_savegame_data_for_report(config, turn)
            if savegame_data is None:
                self.truth_source["turns_without_savegame"].append(turn)
            if savegame_data:
                # Override production data
                if 'production' in savegame_data:
                    for player_id, prod in savegame_data['production'].items():
                        if player_id in player_ids:
                            time_series["trade_production"][turn][player_id] = prod['trade']
                            time_series["food_production"][turn][player_id] = prod['food']
                            time_series["shield_production"][turn][player_id] = prod['shields']

                # Override science data (but NOT techs_known - we use events for that)
                if 'science' in savegame_data:
                    for player_id, sci in savegame_data['science'].items():
                        if player_id in player_ids:
                            time_series["science"][turn][player_id] = sci['science_per_turn']

                # Override the fog-of-war per-player metrics with the server's
                # omniscient view (cities/units/population/territory/treasury/
                # scores/alive/government/wonders).
                if self._override_with_savegame_truth(
                        time_series, turn, savegame_data, player_ids, ruleset,
                        tech_discoveries_by_player):
                    self.truth_source["turns_overridden"] += 1

        # Note: techs_known is already monotonic by construction (cumulative event count)
        return time_series

    def _override_with_savegame_truth(
        self,
        time_series: Dict[str, Dict[int, Dict[int, Any]]],
        turn: int,
        savegame_data: Dict[str, Any],
        player_ids: List[int],
        ruleset: Optional[Dict],
        tech_discoveries_by_player: Dict[int, List[int]],
    ) -> bool:
        """Replace the client-view per-player metrics for one turn with the
        savegame's omniscient values.

        The recorder is a client attached to player 0, so for players 1..N the
        client-derived cities/units/population/territory are the subset the
        observer happened to see, scores are -1 without an embassy and
        treasury is 0 when unknown. The savegame has the complete state.

        Definitions (kept identical to the client-side metrics):
          cities_count          len(city table). Cross-checked vs score.cities
                                and WARNED on mismatch only: the server
                                computes [scoreN] before the turn-change city
                                founding/capture that the save then reflects,
                                so ~0.5% of cells differ by one (rng5001: 6 of
                                ~900); the city table is the value the client
                                also sees for player 0.
          population            sum of city sizes
          units_count           len(unit table)
          military_units_count  savegame_parser.military_unit_count: unit types
                                with attack > 0 (client definition) for
                                classic/civ2civ3, else [scoreN].units
          territory_size        owned tiles from the [map] owner rows
          treasury              gold
          scores                [scoreN].total
          wonders_count         great+small wonders from city improvement bits
                                (needs ruleset improvements + improvement_vector)
          is_alive / government from the [playerN] section

        Returns True if the savegame carried per-player state to apply.
        """
        cities = savegame_data.get('cities') or {}
        units = savegame_data.get('units') or {}
        scores = savegame_data.get('scores') or {}
        gold = savegame_data.get('gold') or {}
        alive = savegame_data.get('alive') or {}
        governments = savegame_data.get('governments') or {}
        barbarian = savegame_data.get('barbarian') or {}
        tile_ownership = savegame_data.get('tile_ownership') or {}
        rulesetdir = savegame_data.get('rulesetdir') or ''

        present = set(cities) | set(scores) | set(alive)
        if not present:
            return False

        wonders = count_wonders_from_savegame(
            cities, savegame_data.get('improvement_vector') or [], ruleset)
        if wonders is not None:
            self.truth_source["wonders_source"] = "savegame city improvements + ruleset genus"

        # Every player present in the savegame gets a row, including ones the
        # client never listed (civilizations is built from the client view);
        # the gate re-checks every savegame player, so skipping any would fail it.
        for pid in sorted(present):
            score_row = scores.get(pid) or {}
            is_barbarian = barbarian.get(pid, 'None') != 'None'

            if pid in scores:
                time_series["scores"][turn][pid] = score_row.get('total', 0)
            if pid in gold:
                time_series["treasury"][turn][pid] = gold[pid]

            if pid in cities:
                city_list = cities[pid]
                n_cities = len(city_list)
                time_series["cities_count"][turn][pid] = n_cities
                time_series["population"][turn][pid] = sum(int(c.get('size', 0)) for c in city_list)
                score_cities = score_row.get('cities')
                # Barbarians are never scored (all [scoreN] fields stay 0)
                if (not is_barbarian and isinstance(score_cities, int)
                        and score_cities != n_cities):
                    self.truth_source["cities_crosscheck_mismatches"].append(
                        {"turn": turn, "player_id": pid,
                         "city_table": n_cities, "score_cities": score_cities})
                    print(f"  WARNING: turn {turn} player {pid}: city table has "
                          f"{n_cities} cities but [score{pid}].cities={score_cities}")

            if pid in units:
                unit_list = units[pid]
                time_series["units_count"][turn][pid] = len(unit_list)
                time_series["military_units_count"][turn][pid] = military_unit_count(
                    unit_list, score_row, is_barbarian, rulesetdir)

            if pid in cities or pid in tile_ownership:
                time_series["territory_size"][turn][pid] = int(tile_ownership.get(pid, 0))

            if wonders is not None and pid in wonders:
                time_series["wonders_count"][turn][pid] = wonders[pid]

            if pid in alive:
                time_series["is_alive"][turn][pid] = bool(alive[pid])
            if pid in governments:
                time_series["government"][turn][pid] = governments[pid]

            # A player the client never listed this turn still needs the
            # event-derived tech count so the row is complete.
            if pid not in time_series["techs_known"][turn]:
                time_series["techs_known"][turn][pid] = sum(
                    1 for t in tech_discoveries_by_player.get(pid, []) if t <= turn)

        return True

    def collect_events(
        self,
        states: Dict[int, Dict],
        data_loader: Any,
        config: Any = None
    ) -> List[Dict[str, Any]]:
        """Collect all game events

        Args:
            states: Dict mapping turn numbers to game states
            data_loader: DataLoader instance
            config: Optional ReportConfig for savegame access

        Returns:
            List of event dictionaries
        """
        detector = EventDetector(data_loader)
        all_events = []

        sorted_turns = sorted(states.keys())
        prev_state = None
        prev_savegame_data = None

        # Track all techs seen so far for each player (to avoid duplicate discoveries)
        all_techs_seen = {}  # {player_id: set of tech_ids}

        for turn in sorted_turns:
            curr_state = states[turn]

            # Try to get savegame data for more complete tech visibility
            curr_savegame_data = None
            if config:
                curr_savegame_data = get_savegame_data_for_report(config, turn)

            # Detect regular events (city, government, diplomatic)
            game_events = detector.detect_city_events(prev_state, curr_state, turn)
            game_events.extend(detector.detect_government_changes(prev_state, curr_state, turn))
            game_events.extend(detector.detect_diplomatic_changes(prev_state, curr_state, turn))

            # Detect wonder completions using savegame data (complete visibility)
            if curr_savegame_data:
                wonder_events = detector.detect_wonder_completions(
                    prev_savegame_data, curr_savegame_data, curr_state, turn,
                    data_loader.ruleset if data_loader else None
                )
                game_events.extend(wonder_events)

            # For tech discoveries, prefer savegame data (complete visibility)
            # Use all_techs_seen to avoid counting same tech multiple times
            if curr_savegame_data:
                curr_techs = curr_savegame_data.get('technologies', {})

                # Get tech names from state if available
                tech_names = {}
                if 'tech' in curr_state:
                    for tech_id, tech_data in curr_state['tech'].items():
                        if isinstance(tech_data, dict):
                            tech_names[tech_id] = tech_data.get('name', f'Tech #{tech_id}')

                # Check each player for genuinely new techs
                for player_id, curr_player_techs in curr_techs.items():
                    if player_id not in all_techs_seen:
                        all_techs_seen[player_id] = set()

                    # Find techs this player knows that we haven't seen before
                    new_techs = curr_player_techs - all_techs_seen[player_id]

                    # Add these techs to the player's seen set
                    all_techs_seen[player_id].update(new_techs)

                    # Create discovery events
                    from ..utils.event_detector import GameEvent
                    for tech_id in new_techs:
                        tech_name = tech_names.get(tech_id, f'Tech #{tech_id}')
                        player_name = detector._get_player_name(curr_state, player_id)

                        game_events.append(GameEvent(
                            turn=turn,
                            event_type='tech_discovered',
                            description=f"{player_name} discovered {tech_name}",
                            player_id=player_id,
                            metadata={
                                'tech_id': tech_id,
                                'tech_name': tech_name
                            }
                        ))
            elif prev_state:
                # Fall back to state-based detection if savegame unavailable
                tech_events = detector.detect_tech_discoveries(prev_state, curr_state, turn)
                game_events.extend(tech_events)

            # Convert GameEvent objects to dictionaries
            for event in game_events:
                event_dict = {
                    "turn": event.turn,
                    "type": event.event_type,
                    "player_id": event.player_id,
                    "description": event.description,
                }

                if event.location:
                    event_dict["location"] = list(event.location)

                if event.metadata:
                    event_dict["metadata"] = event.metadata

                all_events.append(event_dict)

            prev_state = curr_state
            prev_savegame_data = curr_savegame_data

        return all_events

    def collect_snapshots(
        self,
        states: Dict[int, Dict],
        target_turns: List[int],
        config: Any = None,
    ) -> Dict[int, Dict[str, Any]]:
        """Collect snapshot data for specific turns

        Collects scores, rankings, and counts at multiple turns for resolution.
        When ``config`` is given and a savegame exists for the turn, scores,
        rankings and the city/unit counts are taken from the savegame
        (omniscient) instead of the client's fog-of-war view.

        Args:
            states: Dict mapping turn numbers to game states
            target_turns: List of turns to collect snapshots for
            config: Optional ReportConfig (recording_dir) for savegame access

        Returns:
            Dict mapping turn to snapshot data
        """
        snapshots = {}

        for target_turn in target_turns:
            if target_turn not in states:
                continue

            state = states[target_turn]
            players = state.get('player', {})

            # Collect scores and rankings
            scores = {}
            for pid_str, player_info in players.items():
                if isinstance(player_info, dict):
                    pid = int(pid_str)
                    scores[pid] = player_info.get('score', 0)

            # Create rankings
            rankings = []
            for rank, (pid, score) in enumerate(
                sorted(scores.items(), key=lambda x: x[1], reverse=True), 1
            ):
                rankings.append({
                    "rank": rank,
                    "player_id": pid,
                    "score": score
                })

            # Count cities and units per player
            cities = state.get('city', {})
            units = state.get('unit', {})

            cities_count = {}
            units_count = {}
            military_units_count = {}

            for pid in scores.keys():
                cities_count[pid] = len([
                    c for c in cities.values()
                    if isinstance(c, dict) and c.get('owner') == pid
                ])
                units_count[pid] = len([
                    u for u in units.values()
                    if isinstance(u, dict) and u.get('owner') == pid
                ])
                military_units_count[pid] = len([
                    u for u in units.values()
                    if isinstance(u, dict) and u.get('owner') == pid
                    and u.get('type_attack_strength', 0) > 0
                ])

            # World totals
            total_cities = len([c for c in cities.values() if isinstance(c, dict)])
            total_units = len([u for u in units.values() if isinstance(u, dict)])
            total_military_units = len([
                u for u in units.values()
                if isinstance(u, dict) and u.get('type_attack_strength', 0) > 0
            ])
            total_population = sum(
                c.get('size', 0) for c in cities.values() if isinstance(c, dict)
            )

            snapshot = {
                "scores": scores,
                "rankings": rankings,
                "cities_count": cities_count,
                "units_count": units_count,
                "military_units_count": military_units_count,
                "world_totals": {
                    "total_cities": total_cities,
                    "total_units": total_units,
                    "total_military_units": total_military_units,
                    "total_population": total_population
                }
            }

            # Override with the omniscient savegame view when available
            savegame_data = get_savegame_data_for_report(config, target_turn) if config is not None else None
            if savegame_data:
                self._override_snapshot_with_savegame(snapshot, savegame_data)

            snapshots[target_turn] = snapshot

        return snapshots

    @staticmethod
    def _override_snapshot_with_savegame(snapshot: Dict[str, Any], savegame_data: Dict[str, Any]) -> None:
        """Replace a snapshot's scores/rankings/counts with savegame values."""
        sg_cities = savegame_data.get('cities') or {}
        sg_units = savegame_data.get('units') or {}
        sg_scores = savegame_data.get('scores') or {}
        barbarian = savegame_data.get('barbarian') or {}
        rulesetdir = savegame_data.get('rulesetdir') or ''
        present = set(sg_cities) | set(sg_scores)
        if not present:
            return

        for pid in sorted(present):
            score_row = sg_scores.get(pid) or {}
            is_barbarian = barbarian.get(pid, 'None') != 'None'
            if pid in sg_scores:
                snapshot["scores"][pid] = score_row.get('total', 0)
            if pid in sg_cities:
                snapshot["cities_count"][pid] = len(sg_cities[pid])
            if pid in sg_units:
                snapshot["units_count"][pid] = len(sg_units[pid])
                snapshot["military_units_count"][pid] = military_unit_count(
                    sg_units[pid], score_row, is_barbarian, rulesetdir)

        snapshot["rankings"] = [
            {"rank": rank, "player_id": pid, "score": score}
            for rank, (pid, score) in enumerate(
                sorted(snapshot["scores"].items(), key=lambda x: x[1], reverse=True), 1)
        ]
        snapshot["world_totals"] = {
            "total_cities": sum(len(c) for c in sg_cities.values()),
            "total_units": sum(len(u) for u in sg_units.values()),
            "total_military_units": sum(snapshot["military_units_count"].get(pid, 0) for pid in present),
            "total_population": sum(int(c.get('size', 0)) for cl in sg_cities.values() for c in cl),
        }
        snapshot["truth_source"] = TRUTH_SOURCE_SAVEGAME

    def collect_territory_snapshots(
        self,
        states: Dict[int, Dict],
        target_turns: List[int]
    ) -> Dict[int, Dict[str, Any]]:
        """Collect territory map data at specific turns

        Args:
            states: Dict mapping turn numbers to game states
            target_turns: List of turns to collect territory data for

        Returns:
            Dict mapping turn to territory snapshot with tile_owner data
        """
        territory_snapshots = {}

        for turn in target_turns:
            if turn not in states:
                continue

            state = states[turn]
            map_state = state.get('map', {})

            if 'tile_owner' not in map_state:
                continue

            territory_snapshots[turn] = {
                "map_state": {
                    "tile_owner": map_state['tile_owner'],
                    "xsize": map_state.get('xsize', 0),
                    "ysize": map_state.get('ysize', 0),
                }
            }

        return territory_snapshots

    def collect_diplomacy_from_states(
        self,
        states: Dict[int, Dict],
        civilizations: Dict[int, Dict]
    ) -> Dict[str, Any]:
        """Collect diplomatic relationships over time from state files.

        WARNING: This function is BUGGY and should NOT be used for question resolution.
        The state['dipl'] structure only contains the CONTROLLED player's diplomatic
        relationships, not all pairwise relationships. This function incorrectly
        broadcasts a single diplomatic state to all relationship pairs.

        Use collect_diplomacy() instead, which reads from savegame files.

        Extracts diplomatic state (War, Peace, Alliance, etc.) from state['dipl'].

        Args:
            states: Dict mapping turn numbers to game states
            civilizations: Dict of civilization info

        Returns:
            Dict with structure:
            {
                "relations": {
                    "{from_player}_{to_player}": {
                        turn: {
                            "state": str,  # War, Peace, Alliance, etc.
                        }
                    }
                },
                "attitude_thresholds": {...}
            }
        """
        relations = {}
        sorted_turns = sorted(states.keys())
        player_ids = list(civilizations.keys())

        # Attitude thresholds (for reference)
        attitude_thresholds = {
            "worshipful": 820,
            "admiring": 640,
            "enthusiastic": 460,
            "helpful": 280,
            "respectful": 100,
            "neutral": -100,
            "uneasy": -280,
            "uncooperative": -460,
            "hostile": -640,
            "belligerent": -820,
            "genocidal": -1000
        }

        for turn in sorted_turns:
            state = states[turn]
            dipl_data = state.get('dipl', {})

            if not dipl_data:
                continue

            # Each entry in dipl is keyed by player_id, containing their view of diplomacy
            # with other players. The diplomatic_state value maps to DS_* constants.
            for from_player_str, dipl_info in dipl_data.items():
                if not isinstance(dipl_info, dict):
                    continue

                from_player = int(from_player_str)
                if from_player not in player_ids:
                    continue

                # dipl_info contains diplomatic_state for this player's view
                # This is their relationship status with the controlling player
                dipl_state = dipl_info.get('diplomatic_state')

                if dipl_state is not None:
                    state_name = DS_TXT.get(dipl_state, "Unknown")

                    # For each other player, record the relationship
                    # Note: dipl structure is {player_id: {diplomatic_state: N, ...}}
                    # where the key is the other player's ID
                    for to_player in player_ids:
                        if to_player == from_player:
                            continue

                        relation_key = f"{from_player}_{to_player}"

                        if relation_key not in relations:
                            relations[relation_key] = {}

                        relations[relation_key][turn] = {
                            "state": state_name,
                        }

        return {
            "relations": relations,
            "attitude_thresholds": attitude_thresholds
        }

    def collect_diplomacy(
        self,
        states: Dict[int, Dict],
        config: Any,
        civilizations: Dict[int, Dict]
    ) -> Dict[str, Any]:
        """Collect diplomatic relationships over time from savegames.

        This function reads from savegame files which contain complete pairwise
        diplomatic relationships between ALL players. This is the correct method
        for question resolution.

        Extracts both diplomatic state (War, Peace, Alliance, etc.) and
        AI love values (-1000 to 1000) for all player pairs.

        Args:
            states: Dict mapping turn numbers to game states
            config: ReportConfig instance
            civilizations: Dict of civilization info

        Returns:
            Dict with structure:
            {
                "relations": {
                    "{from_player}_{to_player}": {
                        turn: {
                            "state": str,  # War, Peace, Alliance, etc.
                            "love": int    # -1000 to 1000
                        }
                    }
                },
                "attitude_thresholds": {
                    "worshipful": 900,
                    "admiring": 700,
                    ...
                }
            }
        """
        relations = {}
        sorted_turns = sorted(states.keys())
        player_ids = list(civilizations.keys())

        # Attitude thresholds (based on Freeciv source code)
        # MAX_AI_LOVE = 1000, and attitude is divided into 11 levels
        # Each level spans ~182 points (-1000 to 1000 = 2000 range / 11 levels)
        attitude_thresholds = {
            "worshipful": 820,    # >= 820
            "admiring": 640,      # 640 to 819
            "enthusiastic": 460,  # 460 to 639
            "helpful": 280,       # 280 to 459
            "respectful": 100,    # 100 to 279
            "neutral": -100,      # -100 to 99
            "uneasy": -280,       # -280 to -101
            "uncooperative": -460,  # -460 to -281
            "hostile": -640,      # -640 to -461
            "belligerent": -820,  # -820 to -641
            "genocidal": -1000    # < -820
        }

        for turn in sorted_turns:
            # Get savegame data which contains diplomacy
            savegame_data = get_savegame_data_for_report(config, turn)
            if not savegame_data or 'diplomacy' not in savegame_data:
                continue

            diplomacy_data = savegame_data['diplomacy']

            # Process each player's relationships
            for from_player, player_relations in diplomacy_data.items():
                if from_player not in player_ids:
                    continue

                for to_player, relation in player_relations.items():
                    if to_player not in player_ids:
                        continue
                    if from_player == to_player:
                        continue  # Skip self-relationships

                    # Create key for this relationship pair
                    relation_key = f"{from_player}_{to_player}"

                    if relation_key not in relations:
                        relations[relation_key] = {}

                    relations[relation_key][turn] = {
                        "state": relation.get('state', 'Unknown'),
                        "love": relation.get('love', 0),
                        "first_contact_turn": relation.get('first_contact_turn', 0),
                        "embassy": relation.get('embassy', False),
                        "shared_vision": relation.get('shared_vision', False)
                    }

        return {
            "relations": relations,
            "attitude_thresholds": attitude_thresholds
        }

    # Helper methods (replicated from base_section.py)

    def _get_nation_name(self, nation_id: Optional[int], data_loader: Any) -> str:
        """Get nation/civilization name from nation ID

        Args:
            nation_id: Nation ID
            data_loader: DataLoader instance with ruleset data

        Returns:
            Nation name or empty string if not found
        """
        if nation_id is None:
            return ''

        if not data_loader.ruleset or 'nations' not in data_loader.ruleset:
            return ''

        nations = data_loader.ruleset['nations']
        nation_key = str(nation_id)

        if nation_key in nations:
            nation = nations[nation_key]
            return nation.get('adjective') or nation.get('rule_name', '')

        return ''

    def _get_nation_name_from_savegame(
        self,
        nation_identifier: Any,
        data_loader: Any
    ) -> str:
        """Get nation name from savegame nation identifier

        Args:
            nation_identifier: Nation ID (int) or name (str) from savegame
            data_loader: DataLoader instance with ruleset data

        Returns:
            Nation name or empty string if not found
        """
        if not data_loader.ruleset or 'nations' not in data_loader.ruleset:
            return ''

        nations = data_loader.ruleset['nations']

        # Try as numeric ID first
        try:
            nation_id = int(nation_identifier)
            nation_key = str(nation_id)
            if nation_key in nations:
                nation = nations[nation_key]
                return nation.get('adjective') or nation.get('rule_name', '')
        except (ValueError, TypeError):
            pass

        # Try as string name - search for matching rule_name
        nation_str = str(nation_identifier).strip('"')
        for nation_data in nations.values():
            if nation_data.get('rule_name') == nation_str:
                return nation_data.get('adjective') or nation_data.get('rule_name', '')

        return ''
