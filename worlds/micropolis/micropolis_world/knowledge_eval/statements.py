"""Statements about the MicropolisCore engine for the domain-knowledge test.

Drafted by Claude Fable 5 against the engine's source; every fact below was
checked against the fork's code (scan.cpp, evaluate.cpp, simulate.cpp,
zone.cpp, disasters.cpp, tool.cpp) or the cached simulation logs of the
bundled cities in September 2026.

Most statements come in PAIRS: a true statement and a false twin that changes
one thing about it (a number, a direction, a mechanism). The two are always
administered in different prompts (runner.split_halves), so seeing one never
reveals the other. UNPAIRED items have no twin: the disaster types that exist
and the ones that do not, and the HONEYPOTS, mechanics of later SimCity titles
that Micropolis lacks, whose error rate measures contamination.

Difficulty is a subjective rating: 0 (easy), facts anyone who has seen the game
would know; 1 (medium), facts from playing it or reading its manual; 2 (hard),
internals known only from the source. For the honeypots it means "hard to
resist", not "obscure".

Topic groups the statements for the analysis: "engine" is general mechanics and
internals, "cities" the bundled sample cities, and "dynamics" what governs how an
unmanaged city's reported metrics move, the knowledge that could help a model
forecast the world report.
"""

from dataclasses import dataclass

ENGINE = "engine"
CITIES = "cities"
DYNAMICS = "dynamics"


@dataclass(frozen=True)
class Statement:
    text: str
    is_true: bool
    is_honeypot: bool
    difficulty: int
    topic: str = ENGINE
    pair: int | None = None  # index into PAIRS, None for an unpaired statement


# (true statement, false twin, difficulty, topic)
PAIRS: list[tuple[str, str, int, str]] = [
    # ---------- Disasters: the removed one ----------
    (
        "The random airplane-crash disaster from the original SimCity was removed from the open-source Micropolis code.",
        "The tornado disaster from the original SimCity was removed from the open-source Micropolis code.",
        2,
        ENGINE,
    ),  # disasters.cpp doDisasters comment (removed post-9/11)
    # ---------- Disaster triggering & rates ----------
    (
        "Random disasters are triggered by a periodic probability check in the simulation loop.",
        "Random disasters are triggered whenever the city's overall score drops below a threshold.",
        1,
        ENGINE,
    ),  # disasters.cpp doDisasters
    (
        "Random disasters occur more frequently at harder difficulty levels.",
        "Random disasters occur more frequently at easier difficulty levels.",
        1,
        ENGINE,
    ),  # disasters.cpp DisChance = {10*48, 5*48, 60}
    (
        "At the easiest difficulty level, random disasters occur on average roughly once per ten game years.",
        "At the easiest difficulty level, random disasters occur on average roughly once per game year.",
        2,
        ENGINE,
    ),  # DisChance[0] = 10*48 city-time units
    (
        "The random monster disaster is triggered only when the city's average pollution exceeds a threshold.",
        "The random monster disaster is triggered only when the city's crime average exceeds a threshold.",
        2,
        ENGINE,
    ),  # doDisasters monster case gated on pollutionAverage
    (
        "On the hardest difficulty level, each nuclear plant has roughly a one-in-ten-thousand chance of melting down each time its zone is processed.",
        "On the hardest difficulty level, each nuclear plant has roughly a one-in-a-thousand chance of melting down each time its zone is processed.",
        2,
        ENGINE,
    ),  # simulate.cpp meltdownTable = {30000, 20000, 10000}
    (
        "Random disasters can be turned off with a game option.",
        "Random disasters can be disabled only by editing the saved city file.",
        1,
        ENGINE,
    ),  # enableDisasters
    (
        "Meltdowns can only happen to nuclear power plants.",
        "Meltdowns can happen to both coal and nuclear power plants.",
        0,
        ENGINE,
    ),
    # ---------- Disaster impact mechanics ----------
    (
        "Fires can spread to adjacent flammable tiles.",
        "Fires spread only along roads and rail lines.",
        0,
        ENGINE,
    ),
    (
        "When a tile is burning, each neighboring flammable tile has roughly a one-in-eight chance per simulation pass of catching fire.",
        "When a tile is burning, each neighboring flammable tile has roughly a one-in-two chance per simulation pass of catching fire.",
        2,
        ENGINE,
    ),  # simulate.cpp doFire, getRandom16() & 7
    (
        "A tile that finishes burning turns into rubble.",
        "A tile that finishes burning turns into clear, buildable land.",
        1,
        ENGINE,
    ),  # simulate.cpp doFire
    (
        "Fire-station coverage increases the chance that a burning tile burns out quickly.",
        "Fire-station coverage affects only whether new fires ignite, never how quickly existing fires burn out.",
        1,
        ENGINE,
    ),  # doFire reads fireStationEffectMap for the burn-out rate
    (
        "Floods originate at land tiles bordering water.",
        "Floods can begin anywhere on the map, including tiles far from any water.",
        1,
        ENGINE,
    ),  # disasters.cpp makeFlood searches the shoreline
    (
        "Flood water spreads to nearby tiles and later recedes on its own.",
        "Flood water remains on the map permanently until the player bulldozes it.",
        1,
        ENGINE,
    ),  # disasters.cpp doFlood, floodCount
    (
        "A tornado can destroy vehicle sprites, such as ships or trains, that it collides with.",
        "Vehicle sprites such as ships and trains pass through tornadoes without being damaged.",
        2,
        ENGINE,
    ),  # sprite.cpp collision checks
    (
        "An earthquake damages a random number of tiles scattered across the map, turning them to rubble or fire.",
        "An earthquake damages only the tiles along a single fault line drawn across the map.",
        1,
        ENGINE,
    ),  # disasters.cpp makeEarthquake
    (
        "The number of tiles damaged by an earthquake is randomized, on the order of several hundred.",
        "The number of tiles damaged by an earthquake is fixed at exactly 64.",
        2,
        ENGINE,
    ),  # makeEarthquake, getRandom(700) + 300
    (
        "The monster moves toward the map area with the highest pollution.",
        "The monster moves toward the map area with the highest population density.",
        2,
        ENGINE,
    ),  # sprite.cpp, pollutionMaxX/Y
    (
        "A meltdown leaves radioactive tiles that persist on the map long afterward.",
        "Radioactive tiles left by a meltdown can be cleared immediately with the bulldozer.",
        1,
        ENGINE,
    ),  # tool.cpp bulldozer skips RADTILE
    (
        "Each radioactive tile has roughly a one-in-4096 chance per scan of decaying back to clear land.",
        "Each radioactive tile has roughly a one-in-64 chance per scan of decaying back to clear land.",
        2,
        ENGINE,
    ),  # simulate.cpp doRadTile, getRandom16() & 4095
    # ---------- Bundled sample cities: which ship ----------
    # The false twins are plausible siblings of the real file names.
    (
        "Haight is one of the pre-built sample cities that come with the game.",
        "Ashbury is one of the pre-built sample cities that come with the game.",
        1,
        CITIES,
    ),
    (
        "Yokohama is one of the pre-built sample cities that come with the game.",
        "Osaka is one of the pre-built sample cities that come with the game.",
        2,
        CITIES,
    ),
    (
        "Kyoto is one of the pre-built sample cities that come with the game.",
        "Nagoya is one of the pre-built sample cities that come with the game.",
        2,
        CITIES,
    ),
    (
        "Kobe is one of the pre-built sample cities that come with the game.",
        "Sapporo is one of the pre-built sample cities that come with the game.",
        2,
        CITIES,
    ),
    (
        "Kamakura is one of the pre-built sample cities that come with the game.",
        "Nara is one of the pre-built sample cities that come with the game.",
        2,
        CITIES,
    ),
    (
        "Kowloon is one of the pre-built sample cities that come with the game.",
        "Macau is one of the pre-built sample cities that come with the game.",
        2,
        CITIES,
    ),
    (
        "Wetcity is one of the pre-built sample cities that come with the game.",
        "Drycity is one of the pre-built sample cities that come with the game.",
        2,
        CITIES,
    ),
    (
        "Splats is one of the pre-built sample cities that come with the game.",
        "Blobs is one of the pre-built sample cities that come with the game.",
        2,
        CITIES,
    ),
    (
        "Linecity is one of the pre-built sample cities that come with the game.",
        "Gridcity is one of the pre-built sample cities that come with the game.",
        2,
        CITIES,
    ),
    (
        "Deadwood is one of the pre-built sample cities that come with the game.",
        "Tombstone is one of the pre-built sample cities that come with the game.",
        2,
        CITIES,
    ),
    (
        "Saved cities use files with the .cty extension.",
        "Saved cities use files with the .sim extension.",
        1,
        ENGINE,
    ),  # fileio.cpp
    # ---------- Bundled sample cities: what they are like when loaded ----------
    # From the cities' own files, as the engine reports them at turn 0.
    (
        "The bundled city Haight starts with a treasury of one million dollars.",
        "The bundled city Bruce starts with a treasury of one million dollars.",
        2,
        CITIES,
    ),  # haight.cty: $1,000,000; bruce.cty: $281
    (
        "The bundled city Haight starts with more than 200,000 inhabitants.",
        "The bundled city Deadwood starts with more than 200,000 inhabitants.",
        2,
        CITIES,
    ),  # 214,260 against 6,580
    (
        "The bundled city Splats starts with no population at all.",
        "The bundled city Yokohama starts with no population at all.",
        2,
        CITIES,
    ),  # splats.cty has no zones; yokohama.cty starts at 81,380
    (
        "The bundled city Kobe starts with a population above 100,000.",
        "The bundled city Senri starts with a population above 100,000.",
        2,
        CITIES,
    ),  # 153,120 against 20,700
    # ---------- Core mechanics: RCI, power, budget ----------
    (
        "Residential, commercial, and industrial zones each occupy a 3x3 tile footprint.",
        "Residential, commercial, and industrial zones each occupy a 4x4 tile footprint.",
        1,
        ENGINE,
    ),
    (
        "Zones must be connected to electrical power to develop.",
        "Zones develop without electrical power, but grow faster when powered.",
        0,
        ENGINE,
    ),
    (
        "Power is distributed by a flood-fill scan outward from power plants along conductive tiles.",
        "Power reaches any zone within a fixed radius of a power plant, with no wires needed.",
        2,
        ENGINE,
    ),  # power.cpp powerScan
    (
        "In the power model, a coal plant supplies at most 700 units of power.",
        "In the power model, a coal plant supplies at most 500 units of power.",
        2,
        ENGINE,
    ),  # micropolis.h COAL_POWER_STRENGTH = 700
    (
        "In the power model, a nuclear plant supplies at most 2000 units of power.",
        "In the power model, a nuclear plant supplies at most 3000 units of power.",
        2,
        ENGINE,
    ),  # micropolis.h NUCLEAR_POWER_STRENGTH = 2000
    (
        "Raising the tax rate reduces demand for new development.",
        "Raising the tax rate increases demand for new development.",
        1,
        ENGINE,
    ),  # setValves
    (
        "The tax rate can be set between 0% and 20%.",
        "The tax rate can be set between 0% and 10%.",
        1,
        ENGINE,
    ),
    (
        "The default tax rate for a new city is 7%.",
        "The default tax rate for a new city is 5%.",
        1,
        ENGINE,
    ),
    (
        "A new game at the easiest difficulty starts with $20,000.",
        "A new game at the easiest difficulty starts with $10,000.",
        1,
        ENGINE,
    ),  # setGameLevelFunds; $10,000 is the medium level's
    (
        "A new game at the hardest difficulty starts with $5,000.",
        "A new game at the hardest difficulty starts with $2,500.",
        1,
        ENGINE,
    ),
    (
        "Building a road costs $10 per tile.",
        "Building a road costs $20 per tile.",
        1,
        ENGINE,
    ),  # tool.cpp gCostOf; $20 is rail's
    (
        "Fire and police station effectiveness scales with their budget funding level.",
        "Fire and police stations operate at full effectiveness even when their budget funding is cut.",
        1,
        ENGINE,
    ),
    (
        "Roads deteriorate over time when road funding is insufficient.",
        "Roads remain intact permanently regardless of the transportation budget.",
        1,
        ENGINE,
    ),  # simulate.cpp doRoad
    # ---------- Core mechanics: scans and derived maps ----------
    (
        "The pollution map is computed by repeatedly smoothing per-tile pollution contributions.",
        "Pollution affects only the tile that produces it; there is no spreading or smoothing.",
        2,
        ENGINE,
    ),  # scan.cpp
    (
        "Industrial zones contribute to pollution.",
        "Fully powered industrial zones produce no pollution.",
        0,
        ENGINE,
    ),
    (
        "Traffic contributes to pollution.",
        "Traffic lowers nearby pollution levels.",
        1,
        ENGINE,
    ),
    (
        "Crime levels rise with higher population density.",
        "Higher population density lowers a tile's computed crime level.",
        1,
        ENGINE,
    ),  # scan.cpp crimeScan
    (
        "Higher land value lowers a tile's computed crime level.",
        "Higher land value raises a tile's computed crime level.",
        2,
        ENGINE,
    ),  # crimeScan formula
    (
        "Police station coverage reduces crime in the surrounding area.",
        "Police station coverage reduces fire damage in the surrounding area.",
        0,
        ENGINE,
    ),
    (
        "Land value declines with distance from the city center.",
        "Land value rises steadily with distance from the city center.",
        2,
        ENGINE,
    ),  # pollutionTerrainLandValueScan
    (
        "Pollution reduces land value.",
        "Pollution increases land value.",
        1,
        ENGINE,
    ),
    (
        "Traffic generation simulates a random-walk drive that gives up after a maximum distance of about 30 steps.",
        "Traffic generation simulates a random-walk drive that gives up after a maximum distance of about 100 steps.",
        2,
        ENGINE,
    ),  # micropolis.h MAX_TRAFFIC_DISTANCE = 30
    # ---------- Engine internals ----------
    (
        "The city map is 120 tiles wide and 100 tiles tall.",
        "The city map is 128 tiles wide and 128 tiles tall.",
        2,
        ENGINE,
    ),  # micropolis.h WORLD_W / WORLD_H
    (
        "Each pass of the simulator is divided into 16 phases.",
        "Each pass of the simulator is divided into 8 phases.",
        2,
        ENGINE,
    ),  # simulate.cpp simulate() switch
    (
        "48 units of city time correspond to one game year.",
        "12 units of city time correspond to one game year.",
        2,
        ENGINE,
    ),  # cityYear computation
    (
        "A newly generated city starts in the year 1900.",
        "A newly generated city starts in the year 1950.",
        1,
        ENGINE,
    ),  # startingYear
    (
        "The engine tracks history graphs for residential, commercial, and industrial population as well as cash flow, crime, and pollution.",
        "The engine tracks history graphs for land value and traffic density.",
        1,
        ENGINE,
    ),  # resHist, comHist, indHist, moneyHist, crimeHist, pollutionHist
    (
        "The city evaluation produces a city score on a scale up to 1000.",
        "The city evaluation produces a city score on a scale up to 100.",
        2,
        ENGINE,
    ),  # evaluate.cpp doScore
    (
        "The number of citizens who approve of the mayor depends on the city score.",
        "The evaluation includes the percentage of citizens who approve of the city council.",
        1,
        ENGINE,
    ),  # evaluate.cpp doVotes
    (
        "Hospitals and churches are placed by the simulator on residential land rather than built by the player.",
        "Hospitals must be placed by the player using a dedicated hospital tool.",
        2,
        ENGINE,
    ),  # zone.cpp makeHospital
    (
        "Once the residential population is large enough, the game demands that a stadium be built.",
        "Once the residential population is large enough, the game demands that an airport be built.",
        1,
        ENGINE,
    ),  # message.cpp; the airport is the commercial demand
    (
        "Airports spawn aircraft sprites that fly over the city.",
        "Airports launch aircraft only after the player builds a control tower.",
        1,
        ENGINE,
    ),
    (
        "A helicopter sprite reports areas of heavy traffic.",
        "A helicopter sprite reports areas of high crime.",
        2,
        ENGINE,
    ),  # sprite.cpp doCopterSprite
    (
        "The seaport spawns a ship sprite that travels the waterways.",
        "A seaport functions only when placed adjacent to a rail line.",
        1,
        ENGINE,
    ),
    (
        "The engine includes a terrain generator that creates random maps with rivers.",
        "The terrain generator can only produce island maps.",
        1,
        ENGINE,
    ),  # generate.cpp; terrainCreateIsland is one option
    (
        "An auto-bulldoze option lets building tools clear trees and rubble automatically.",
        "An auto-bulldoze option automatically levels existing developed zones when you build over them.",
        1,
        ENGINE,
    ),  # autoBulldoze flag
    # ---------- Dynamics of an unmanaged city ----------
    # What moves the metrics the world report shows: the knowledge that could
    # help a model forecast them.
    (
        "The average pollution, crime and land value the engine reports are means over map cells whose values range from 0 to 255.",
        "The average pollution, crime and land value the engine reports are means over map cells whose values range from 0 to 100.",
        2,
        DYNAMICS,
    ),  # scan.cpp: averages of byte-valued maps
    (
        "The average traffic figure is the mean traffic density over developed land, multiplied by 2.4.",
        "The average traffic figure is the mean traffic density over developed land, divided by 2.",
        2,
        DYNAMICS,
    ),  # evaluate.cpp getTrafficAverage
    (
        "When nobody builds new zones, a city's population can grow only by its existing zones becoming denser.",
        "When demand is high, zones expand on their own into neighboring empty land.",
        1,
        DYNAMICS,
    ),  # zone.cpp: development changes a zone's density tile, never its footprint
    (
        "The reported population is derived from the residential, commercial and industrial zone populations by a fixed formula, not by counting individual citizens.",
        "The reported population counts simulated individual citizens moving between homes and workplaces.",
        1,
        DYNAMICS,
    ),  # evaluate.cpp getPopulation
    (
        "The reported population is always a multiple of 20.",
        "The reported population is always a multiple of 50.",
        2,
        DYNAMICS,
    ),  # getPopulation: (resPop + 8*(comPop + indPop)) * 20
    (
        "The census that updates the city's history runs every 4 units of city time, twelve times per game year.",
        "The census that updates the city's history runs every 48 units of city time, once per game year.",
        2,
        DYNAMICS,
    ),  # simulate.cpp CENSUS_FREQUENCY_10 = 4
    (
        "Average land value is recomputed by a periodic scan from pollution and distance to the center, so it changes even when nothing is built or demolished.",
        "Land value changes only when the player builds or demolishes something.",
        1,
        DYNAMICS,
    ),  # scan.cpp pollutionTerrainLandValueScan
    (
        "Zones that lose electrical power lose population over time.",
        "Zones that lose electrical power keep their population but stop growing.",
        1,
        DYNAMICS,
    ),  # zone.cpp doResidential: zscore = -500 without power, then doResOut
    (
        "The city score is computed from a table of citizen problems and the residential, commercial and industrial demand, scaled to a 0 to 1000 range.",
        "The city score is computed only from the population and the treasury balance.",
        2,
        DYNAMICS,
    ),  # evaluate.cpp doScore
    (
        "Rubble left by fires and disasters stays on the map until it is bulldozed.",
        "Rubble left by fires and disasters clears on its own after about a game year.",
        1,
        DYNAMICS,
    ),  # simulate.cpp mapScan skips tiles below FLOOD
    (
        "With road funding at 100%, roads never deteriorate.",
        "Roads deteriorate with age even when road funding is at 100%.",
        2,
        DYNAMICS,
    ),  # simulate.cpp doRoad: decay only below 15/16 of MAX_ROAD_EFFECT
    (
        "The citizen problems the evaluation counts are crime, pollution, housing costs, taxes, traffic, unemployment and fire.",
        "The citizen problems the evaluation counts include education and health care.",
        2,
        DYNAMICS,
    ),  # micropolis.h CVP_* enum
    (
        "Traffic density on a road tile decays over time once trips stop passing through it.",
        "Traffic density on a road tile stays at its peak until the road is bulldozed.",
        1,
        DYNAMICS,
    ),  # simulate.cpp decTrafficMap every pass
]

# (statement, difficulty, topic): true statements with no false twin.
UNPAIRED_TRUE: list[tuple[str, int, str]] = [
    ("Fires are one of the disasters that can occur in the game.", 0, ENGINE),
    ("Floods are one of the disasters that can occur in the game.", 0, ENGINE),
    ("Tornadoes are one of the disasters that can occur in the game.", 0, ENGINE),
    ("Earthquakes are one of the disasters that can occur in the game.", 0, ENGINE),
    (
        "An attack by a giant monster is one of the disasters that can occur in the game.",
        0,
        ENGINE,
    ),
    (
        "A nuclear power plant meltdown is one of the disasters that can occur in the game.",
        0,
        ENGINE,
    ),
    (
        "Shipwrecks are one of the accidents that can occur in the game.",
        1,
        ENGINE,
    ),  # sprite.cpp ship crash + 'Shipwreck' message
    ("Pollution is an in-game mechanic tracked per map location.", 0, ENGINE),
    ("Traffic is an in-game mechanic tracked per map location.", 0, ENGINE),
    ("Crime is an in-game mechanic tracked per map location.", 0, ENGINE),
]

# (statement, difficulty, topic): false statements with no true twin, mostly
# disasters the engine does not have.
UNPAIRED_FALSE: list[tuple[str, int, str]] = [
    (
        "Volcanic eruptions are one of the disasters that can occur in the game.",
        0,
        ENGINE,
    ),
    ("Hurricanes are one of the disasters that can occur in the game.", 0, ENGINE),
    ("Blizzards are one of the disasters that can occur in the game.", 0, ENGINE),
    ("Landslides are one of the disasters that can occur in the game.", 0, ENGINE),
    (
        "An invasion by alien spacecraft is one of the disasters that can occur in the game.",
        1,
        ENGINE,
    ),
    ("A chemical spill is one of the disasters that can occur in the game.", 1, ENGINE),
    (
        "Bridge collapses are one of the accidents that can occur in the game.",
        1,
        ENGINE,
    ),
    (
        "Tornadoes strike a single fixed location and dissipate without moving.",
        1,
        ENGINE,
    ),  # the tornado is a moving sprite
]

# (statement, difficulty): mechanics absent from Micropolis, from SimCity 2000
# and later. All false; tracked as their own subset.
HONEYPOTS: list[tuple[str, int]] = [
    ("Water pipes must be laid underground to supply zones with water.", 1),
    ("A separate water-coverage value is tracked per map location.", 0),
    ("Subway lines can be built as an underground alternative to roads.", 1),
    ("Highways can be built that carry more traffic than regular roads.", 2),
    ("Bus depots can be built to take traffic off the roads.", 1),
    (
        "The player can enact city ordinances, such as a pollution tax, to shape the city.",
        1,
    ),
    ("Neighboring cities can be connected to buy and sell electricity.", 1),
    ("Arcologies become available once the city grows large enough.", 0),
    ("Terrain can be raised and lowered with terraforming tools during play.", 0),
    ("Residential zones can be zoned separately as light or dense.", 2),
    ("Farmland can be zoned as a separate agricultural zone type.", 1),
    ("Schools and colleges are buildings the player can construct.", 1),
    ("An education level is tracked per map location.", 0),
    ("A garbage level is tracked per map location.", 0),
    ("Landfills must be zoned to dispose of the city's garbage.", 1),
    ("Prisons can be built to reduce crime.", 1),
    ("Hydroelectric dams can be built on waterfalls to generate power.", 1),
    ("Fusion power plants become available in later game years.", 1),
    (
        "The player can set separate tax rates for residential, commercial and industrial zones.",
        2,
    ),
    (
        "A rewards system grants special buildings, such as a mayor's house, at population milestones.",
        1,
    ),
]


def all_statements() -> list[Statement]:
    """Every statement, unshuffled: pairs first, then the unpaired, then honeypots."""
    out: list[Statement] = []
    for i, (true_text, false_text, difficulty, topic) in enumerate(PAIRS):
        out.append(Statement(true_text, True, False, difficulty, topic, i))
        out.append(Statement(false_text, False, False, difficulty, topic, i))
    out += [Statement(t, True, False, d, topic) for t, d, topic in UNPAIRED_TRUE]
    out += [Statement(t, False, False, d, topic) for t, d, topic in UNPAIRED_FALSE]
    out += [Statement(t, False, True, d, ENGINE) for t, d in HONEYPOTS]
    return out
