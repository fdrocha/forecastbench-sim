"""
Savegame modification for world forking experiments.

Modifies FreeCiv savegame files to inject events/changes before reloading.
This enables conditional forecasting: "P(B | A happened)" with ground-truth from simulation.
"""

import lzma
import re
import tempfile
import subprocess
from pathlib import Path
from typing import Optional


class SavegameModifier:
    """Modify FreeCiv savegame files to inject events."""

    def __init__(self, savegame_path: str):
        """
        Initialize with path to a compressed savegame (.sav.xz).

        Args:
            savegame_path: Path to the .sav.xz file
        """
        self.savegame_path = Path(savegame_path)
        self.content: Optional[str] = None
        self._load()

    def _load(self) -> None:
        """Load and decompress the savegame (.sav.xz, .sav.zst, or plain .sav)."""
        name = self.savegame_path.name
        if name.endswith('.zst'):
            import zstandard
            with open(self.savegame_path, 'rb') as f:
                dctx = zstandard.ZstdDecompressor()
                with dctx.stream_reader(f) as reader:
                    self.content = reader.read().decode('utf-8')
        elif name.endswith('.xz'):
            with lzma.open(self.savegame_path, 'rt') as f:
                self.content = f.read()
        else:
            with open(self.savegame_path, 'r') as f:
                self.content = f.read()

    def save(self, output_path: Optional[str] = None) -> str:
        """
        Save the modified savegame, compressed to match the output extension
        (.xz via lzma, .zst via zstandard, otherwise uncompressed).

        Args:
            output_path: Where to save. If None, overwrites original.

        Returns:
            Path to saved file.
        """
        if output_path is None:
            output_path = str(self.savegame_path)

        output_path = Path(output_path)
        name = output_path.name
        if name.endswith('.zst'):
            import zstandard
            cctx = zstandard.ZstdCompressor()
            with open(output_path, 'wb') as f:
                f.write(cctx.compress(self.content.encode('utf-8')))
        elif name.endswith('.xz'):
            with lzma.open(output_path, 'wt') as f:
                f.write(self.content)
        else:
            with open(output_path, 'w') as f:
                f.write(self.content)

        return str(output_path)

    def save_uncompressed(self, output_path: str) -> str:
        """Save without compression (for debugging)."""
        with open(output_path, 'w') as f:
            f.write(self.content)
        return output_path

    # -------------------------------------------------------------------------
    # Player modifications
    # -------------------------------------------------------------------------

    def set_player_name(self, player_id: int, name: str) -> None:
        """
        Set a player's name AND connection identity.

        Rewrites `name=`, `username=` and `ranked_username=` in the
        [player{N}] section. Rewriting the username fields is load-bearing for
        forking: on `/load`, the server auto-reattaches connections to players
        by username, so the fork client (connected under the fork username)
        attaches to player0 without any `/take`+`/aitoggle` dance — which is
        what used to wipe the player's AI research goal (goal_name=A_UNSET).

        Args:
            player_id: Player number (0-indexed)
            name: New player name / username
        """
        section_start = self.content.find(f'[player{player_id}]\n')
        if section_start == -1:
            section_start = self.content.find(f'[player{player_id}]')
        if section_start == -1:
            raise ValueError(f"set_player_name: no [player{player_id}] section found")
        section_end = self.content.find('\n[', section_start + 1)
        section_end = len(self.content) if section_end == -1 else section_end
        section = self.content[section_start:section_end]

        section, n_name = re.subn(
            r'^name="[^"]*"', f'name="{name}"', section, count=1, flags=re.MULTILINE)
        if n_name == 0:
            raise ValueError(
                f"set_player_name: no name= field in [player{player_id}] section")
        # username= / ranked_username= exist in server-written saves; rewrite
        # them so load auto-reattach targets this player.
        section, _ = re.subn(
            r'^username="[^"]*"', f'username="{name}"', section, count=1,
            flags=re.MULTILINE)
        section, _ = re.subn(
            r'^ranked_username="[^"]*"', f'ranked_username="{name}"', section,
            count=1, flags=re.MULTILINE)

        self.content = (self.content[:section_start] + section
                        + self.content[section_end:])

    def set_player_gold(self, player_id: int, gold: int) -> None:
        """
        Set a player's gold amount.

        Args:
            player_id: Player number (0-indexed)
            gold: New gold amount
        """
        pattern = rf'(\[player{player_id}\].*?gold=)\d+'
        replacement = rf'\g<1>{gold}'
        self.content = re.sub(pattern, replacement, self.content, flags=re.DOTALL)

    def add_player_gold(self, player_id: int, amount: int) -> None:
        """
        Add gold to a player's current amount.

        Args:
            player_id: Player number (0-indexed)
            amount: Gold amount to add (can be negative)
        """
        current = self.get_player_gold(player_id)
        self.set_player_gold(player_id, current + amount)

    def set_player_government(self, player_id: int, government: str) -> None:
        """
        Set a player's government type.

        Args:
            player_id: Player number
            government: Government name (e.g., "Republic", "Monarchy", "Democracy")
        """
        pattern = rf'(\[player{player_id}\].*?government_name=")[^"]*"'
        replacement = rf'\g<1>{government}"'
        self.content = re.sub(pattern, replacement, self.content, flags=re.DOTALL)

    def grant_player_tech(self, player_id: int, tech_id: int) -> None:
        """
        Grant a technology to a player.

        Technologies are stored as a binary string where each position
        represents whether that tech is known (1) or not (0).

        Raises:
            ValueError: If no tech row/field matched for the player (silent
                no-ops here previously corrupted conditional ground truth).

        Args:
            player_id: Player number
            tech_id: Technology ID to grant
        """
        # Newer save format: techs live in the [research] table, one row per
        # research number (== player id without team research), with a known-
        # techs count in column 3 and a "done" bitstring as the last field:
        #   0,"The Republic",8,0,2,"",30,"The Republic",0,"1010...01"
        research_idx = self.content.find('[research]')
        if research_idx != -1:
            row_pattern = re.compile(
                rf'^({player_id},"[^"]*",)(\d+)(,.*,")([01]+)("\s*)$', re.MULTILINE)

            def replace_row(match):
                head, count, mid, bits, tail = match.groups()
                if tech_id >= len(bits) or bits[tech_id] == '1':
                    return match.group(0)
                new_bits = bits[:tech_id] + '1' + bits[tech_id + 1:]
                return f'{head}{int(count) + 1}{mid}{new_bits}{tail}'

            section_end = self.content.find('\n[', research_idx + 1)
            section_end = len(self.content) if section_end == -1 else section_end
            section = self.content[research_idx:section_end]
            new_section, n = row_pattern.subn(replace_row, section)
            if n == 0:
                raise ValueError(
                    f"grant_player_tech: no [research] row matched for player {player_id}")
            self.content = (self.content[:research_idx] + new_section
                            + self.content[section_end:])
            return

        # Old save format: per-player inventions bitstring.
        pattern = rf'(\[player{player_id}\].*?research="inventions",")[01]*"'

        def replace_tech(match):
            prefix = match.group(1)
            # Extract the current tech string
            full_match = match.group(0)
            tech_string = full_match.split('"')[-2]

            # Convert to list, modify, convert back
            tech_list = list(tech_string)
            if tech_id < len(tech_list):
                tech_list[tech_id] = '1'
            tech_string = ''.join(tech_list)

            return f'{prefix}{tech_string}"'

        new_content, n = re.subn(pattern, replace_tech, self.content, flags=re.DOTALL)
        if n == 0:
            raise ValueError(
                f"grant_player_tech: no inventions field matched for player {player_id}")
        self.content = new_content

    # -------------------------------------------------------------------------
    # RNG state mutation (for Monte Carlo rollouts)
    # -------------------------------------------------------------------------

    @staticmethod
    def _fc_srand_state(seed: int) -> tuple:
        """Reproduce Freeciv's ``fc_srand(seed)`` in Python.

        Returns ``(v, j, k, x)`` matching the global ``rand_state`` Freeciv
        would hold after ``fc_srand(seed)``. Mirrors ``utility/rand.c``
        bit-for-bit: linear congruential init, then a 10000-iteration
        warm-up via the Mitchell-Moore additive generator with
        ``size = MAX_UINT32``.
        """
        MASK = 0xFFFFFFFF
        v = [0] * 56
        v[0] = seed & MASK
        for i in range(1, 56):
            v[i] = (3 * v[i - 1] + 257) & MASK
        j, k, x = 0, 31, 55

        # Heat-up: 10000 iterations of fc_rand(MAX_UINT32).
        # With size == MAX_UINT32, divisor == 1 and max == MAX_UINT32 - 1,
        # so the rejection branch only fires when v[j]+v[k] == MASK exactly.
        for _ in range(10000):
            while True:
                new_rand = (v[j] + v[k]) & MASK
                x = (x + 1) % 56
                j = (j + 1) % 56
                k = (k + 1) % 56
                v[x] = new_rand
                if new_rand <= MASK - 1:
                    break
        return v, j, k, x

    def set_rng_from_seed(self, seed: int) -> None:
        """Overwrite the savegame's ``[random]`` block with the state Freeciv
        would have after ``fc_srand(seed)``.

        Two Freeciv-only callers of ``fc_srand`` produce game RNG state:
        ``init_game_seed`` at game start, and (after loading) restoration of
        the saved table. By writing a ``fc_srand``-equivalent state directly
        into the savegame, we get a deterministic, seed-controlled rollout
        from this saved turn — without patching the server.
        """
        v, j, k, x = self._fc_srand_state(seed)

        def _table_line(words) -> str:
            # Freeciv writes each word as %8x (lowercase, space-padded, no
            # leading zeros), space-separated, the whole thing in quotes.
            return '"' + ' '.join(f'{w:8x}' for w in words) + '"'

        new_block_lines = [
            '[random]',
            'saved=TRUE',
            f'index_J={j}',
            f'index_K={k}',
            f'index_X={x}',
        ]
        for t in range(8):
            words = v[t * 7:(t + 1) * 7]
            new_block_lines.append(f'table{t}={_table_line(words)}')
        new_block = '\n'.join(new_block_lines)

        # Replace the existing [random] block (up to the next [section] header).
        pattern = r'\[random\].*?(?=\n\[)'
        new_content, n = re.subn(pattern, new_block, self.content, count=1, flags=re.DOTALL)
        if n != 1:
            raise RuntimeError("Failed to locate [random] block in savegame")
        self.content = new_content

    def clear_rng_state(self) -> None:
        """Make the engine generate a FRESH game seed when this save loads.

        Marks the saved ``[random]`` table as not-saved (``saved=FALSE``) so
        the server ignores it on load, and zeroes the ``gameseed`` setting so
        ``init_game_seed`` draws a fresh seed instead of replaying the
        recorded one. This is the default per-fork reseed: without it every
        fork resumes the identical saved RNG table and the fork ensemble
        collapses to a near point mass.
        """
        new_content, n = re.subn(
            r'(\[random\]\s*\nsaved=)TRUE', r'\g<1>FALSE', self.content, count=1)
        if n != 1:
            # saved=FALSE already means the table is ignored on load.
            if not re.search(r'\[random\]\s*\nsaved=FALSE', self.content):
                raise RuntimeError(
                    "clear_rng_state: no [random] block with saved= flag found")
        else:
            self.content = new_content

        # Zero the recorded gameseed so the engine generates a fresh one.
        # Settings rows look like: "gameseed",1000,1000,"Changed" — only the
        # first numeric column (the live value) is rewritten.
        new_content, n = re.subn(
            r'("gameseed",)\d+', r'\g<1>0', self.content, count=1)
        if n == 1:
            self.content = new_content
        # If no gameseed settings row exists the server default (0) already
        # generates a fresh seed, so nothing to do.

    # -------------------------------------------------------------------------
    # Game state queries
    # -------------------------------------------------------------------------

    def get_player_gold(self, player_id: int) -> int:
        """Get current gold for a player."""
        pattern = rf'\[player{player_id}\].*?gold=(\d+)'
        match = re.search(pattern, self.content, flags=re.DOTALL)
        if match:
            return int(match.group(1))
        return 0

    def get_turn(self) -> int:
        """Get the current turn number."""
        match = re.search(r'turn=(\d+)', self.content)
        if match:
            return int(match.group(1))
        return 0

    def get_player_count(self) -> int:
        """Count number of players in the savegame."""
        return len(re.findall(r'\[player\d+\]', self.content))

    # -------------------------------------------------------------------------
    # Docker integration
    # -------------------------------------------------------------------------

    def upload_to_docker(self, username: str, container_name: str = 'freeciv-web') -> str:
        """
        Upload modified savegame to Docker container.

        Args:
            username: FreeCiv username (determines save directory)
            container_name: Docker container name

        Returns:
            The savegame name (without path) for use with /load command
        """
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.sav.xz', delete=False) as f:
            temp_path = f.name

        self.save(temp_path)

        # Determine docker path
        savegame_name = self.savegame_path.name
        docker_path = f'/var/lib/tomcat10/webapps/data/savegames/{username}/{savegame_name}'

        # Copy to container
        subprocess.run(
            ['docker', 'cp', temp_path, f'{container_name}:{docker_path}'],
            check=True
        )

        # Clean up temp file
        Path(temp_path).unlink()

        # Return the savegame name (without extension) for /load command
        return savegame_name.replace('.sav.xz', '')
