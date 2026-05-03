from os.path import join

from pandas import DataFrame, read_csv

from multabench.datasets.curation_objects import CuratedTarget
from multabench.datasets.objects import SupervisedTask

'''
Dataset Name: jeradrose/hearthstone-cards/
====
Examples: 2810
====
URL: https://www.kaggle.com/jeradrose/hearthstone-cards
====

Hearthstone Cards Dataset
Describes collectable cards from the game Hearthstone, with features covering gameplay stats, categorical traits, and
rich textual descriptions. The prediction target is the player class to which a card belongs (10 categories), with a
notable class imbalance. Neutral cards dominate, while other classes have significantly fewer samples. This natural
imbalance reflects the game design and poses a realistic challenge for classification tasks. Text fields like text (card
effects) and flavor (lore snippets) provide valuable semantic signals, making this dataset well-suited to benchmark how
textual features can improve tabular model performance, especially for minority classes.


Description:
This dataset contains data for the entire collection of cards for Hearthstone, the popular online card game by Blizzard. Launching to the public on March 11, 2011 after being under development for almost 5 years, Hearthstone has gained popularity as a freemium game, launching into eSports across the globe, and the source of many Twitch channels.

The data in this dataset was extracted from hearthstonejson.com, and the documentation for all the data can be found on the cards.json documentation page.

The original data was extracted from the actual card data files used in the game, so all of the data should be here, enabling explorations like:

Card strengths and weaknesses
Card strengths relative to cost and rarity
Comparisons across player classes, bosses, and sets
Whether a set of optimal cards can be determined per class
The cards can be explored in one of four ways:

cards.json: The raw JSON pulled from hearthstonejson.com
cards_flat.csv: A flat CSV containing a row for each card, and any n:m data stored as arrays in single fields
database.sqlite: A SQLite database containing relational data of the cards
cards.csv, mechanics.csv, dust_costs.csv, play_requirements.csv, and entourages.csv: the normalized data in CSV format.
This dataset will be updated as new releases and expansions are made to Hearthstone.

Currently, any localized string values are in en-us, but I may look into adding other languages if the demand seems to be there.
====
Target Variable: playerClass (object, 10 distinct): ['NEUTRAL', 'DRUID', 'WARRIOR', 'HUNTER', 'MAGE', 'ROGUE', 'SHAMAN', 'WARLOCK', 'PALADIN', 'PRIEST']
====
Features:

type (object, 6 distinct): ['MINION', 'SPELL', 'ENCHANTMENT', 'HERO_POWER', 'HERO', 'WEAPON']
name (object, 2195 distinct): ['Jade Golem', 'Smuggling', 'Enraged', 'Nefarian', 'Fate', 'Goldthorn', 'Hungry Naga', 'Whelp', 'Living Bomb', 'Decimate']
set (object, 17 distinct): ['EXPERT1', 'TB', 'GANGS', 'LOE', 'BRM', 'KARA', 'OG', 'CORE', 'TGT', 'GVG']
text (object, 1875 distinct): ['<b>Taunt</b>', 'Increased Attack.', '+1/+1.', 'Increased stats.', '+2/+2.', '<b>Charge</b>', '+3 Attack.', '<b>Spell Damage +1</b>', '<b>Stealth</b>', '<b>Windfury</b>']
cost (float64, 15 distinct): ['2.0', '0.0', '1.0', '3.0', '4.0', '5.0', '6.0', '10.0', '7.0', '8.0']
attack (float64, 31 distinct): ['2.0', '1.0', '3.0', '4.0', '5.0', '0.0', '6.0', '7.0', '8.0', '10.0']
health (float64, 42 distinct): ['2.0', '4.0', '1.0', '3.0', '5.0', '30.0', '6.0', '7.0', '8.0', '10.0']
rarity (object, 5 distinct): ['COMMON', 'RARE', 'LEGENDARY', 'EPIC', 'FREE']
flavor (object, 1056 distinct): ['They have an uneasy rivalry with the Blood Paladins.', ...]
race (object, 7 distinct): ['BEAST', 'MECHANICAL', 'DEMON', 'DRAGON', 'MURLOC', 'PIRATE', 'TOTEM']
how_to_earn (object, 29 distinct): ['Unlocked at Level 1.', ...]
how_to_earn_golden (object, 80 distinct): ['Crafting unlocked in the Hall of Explorers, in the League of Explorers adventure.', ...]
targeting_arrow_text (object, 41 distinct): ['Deal 1 damage.', '<b>Silence</b> a minion.', 'Give +2 Attack.', 'Deal 3 damage.', 'Deal 4 damage.', 'Restore 2 Health.', 'Give +1/+1.', 'Deal 6 damage.', 'Deal 2 damage.', 'Return a minion to your hand.']
faction (object, 2 distinct): ['ALLIANCE', 'HORDE']
durability (float64, 7 distinct): ['2.0', '3.0', '4.0', '5.0', '8.0', '6.0', '1.0']
'''

def load_df(dir_path: str) -> DataFrame:
    df_path = join(dir_path, 'cards.csv')
    df = read_csv(df_path, encoding="utf-8", encoding_errors="replace")
    # Removing DREAM which has only 5 appearances and DEATHKNIGHT with a single one (all others have 100+)
    too_rare_class_categories = ['DREAM', 'DEATHKNIGHT']
    df = df[df['playerClass'].apply(lambda c: c not in too_rare_class_categories)]
    return df


CONTEXT = "Card game Hearthstone: Heroes of Warcraft"
TARGET = CuratedTarget(raw_name="playerClass", task_type=SupervisedTask.MULTICLASS)
FEATURES = []
COLS_TO_DROP = ['card_id',
                # Constant
                'collectible']
IMAGE_FOLDER = None
LOADING_FUNC = load_df
PROCESSING_FUNC = None
