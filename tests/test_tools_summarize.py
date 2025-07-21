import sys
sys.path.append("./")
from src.tools.summerize import MapReduceSummerizer, IterativeRefineSummerizer


text_1 ='''
Emma Caldwell had always thought that apartments held their breath when dusk slipped in, that the walls themselves waited for darkness the way lungs wait for air. On Wednesday, November 4, 2025, the city of Seattle moved from gold to pewter at exactly 5:06 p.m., the first streetlamps igniting halos through a gauze of drizzle. By 5:20 p.m. Emma’s third-floor walk-up on Olive Way glowed with the amber light of a single floor lamp, the edges of its parquet floor flickering like coals beneath its shade. At 5:42 p.m. a Le Creuset pot began to simmer on the gas range, perfume of red wine and thyme spiraling upward. Soft jazz drifted from a Bluetooth speaker at 5:55 p.m., the bassline threading through the clatter of a wooden spoon against cast iron .
At 6:14 p.m., Daniel Kim across the hall drew the first note of a Bach cello suite, the sound tunneling through walls older than the Second World War. Downstairs, at precisely 6:18 p.m., Mrs. Alvarez coaxed her cat Sir Isaac Mewton to accept supper, the can lid pinging in her kitchen like a small brass bell. Somewhere around 6:40 p.m. Emma saved the last edits on a freelance article titled “Comfort Foods for Darker Days,” closed her laptop, and let herself luxuriate in the slow-rolling cookery. She did not see—could not have seen—the notification in her tenant portal at 6:47 p.m. that the superintendent had once again postponed replacing the building’s century-old locks
'''

def test_split_text():
    summerize = MapReduceSummerizer(text_1)

    chunks = summerize.get_documents()

    assert len(chunks) == 2

def test_map_reduce_decomposed():
    summerizer = MapReduceSummerizer(text_1)
    mapping = summerizer.map()

    assert len(mapping) == len(summerizer.get_documents())

    final_summary = summerizer.reduce(mapping)

    print(final_summary)


def test_refine_summary_1():
    summerizer = IterativeRefineSummerizer.from_recursive_splitter(text_1)

    summary = summerizer.run()

    print(summary)




