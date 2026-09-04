"""
Dataset Name: MUL_TEXT_HEARTHSTONE_CARDS
====
Examples: 2810
====
URL: https://www.kaggle.com/jeradrose/hearthstone-cards
====
Target Variable: playerClass (object, 10 distinct): ['NEUTRAL', 'DRUID', 'WARRIOR', 'HUNTER', 'MAGE', 'ROGUE', 'SHAMAN', 'WARLOCK', 'PALADIN', 'PRIEST']
====
Features:

type (object, 6 distinct): ['MINION', 'SPELL', 'ENCHANTMENT', 'HERO_POWER', 'HERO', 'WEAPON']
name (object, 2195 distinct): ['Jade Golem', 'Smuggling', 'Enraged', 'Nefarian', 'Fate', 'Goldthorn', 'Hungry Naga', 'Whelp', 'Living Bomb', 'Decimate']
set (object, 17 distinct): ['EXPERT1', 'TB', 'GANGS', 'LOE', 'BRM', 'KARA', 'OG', 'CORE', 'TGT', 'GVG']
text (object, 1875 distinct, 15.3% missing): ['<b>Taunt</b>', 'Increased Attack.', '+1/+1.', 'Increased stats.', '+2/+2.', '<b>Charge</b>', '+3 Attack.', '<b>Spell Damage +1</b>', '<b>Stealth</b>', '<b>Windfury</b>']
cost (float64, 15 distinct, 23.1% missing): ['2.0', '0.0', '1.0', '3.0', '4.0', '5.0', '6.0', '10.0', '7.0', '8.0']
attack (float64, 31 distinct, 55.3% missing): ['2.0', '1.0', '3.0', '4.0', '5.0', '0.0', '6.0', '7.0', '8.0', '10.0']
health (float64, 42 distinct, 51.4% missing): ['2.0', '4.0', '1.0', '3.0', '5.0', '30.0', '6.0', '7.0', '8.0', '10.0']
rarity (object, 5 distinct, 50.2% missing): ['COMMON', 'RARE', 'LEGENDARY', 'EPIC', 'FREE']
flavor (object, 1056 distinct, 62.4% missing): ["It's like putting racing stripes and a giant spoiler on your hero power.", "Rejected names: Forty-Foot Faceless, Big ol' No-face, Huge Creature Sans Face, Teddy.", 'HATES being called "the wonder twins".', 'If she threatens to "moon" you, it\'s not what you think.', "He's the reason the First Bank of Gadgetzan has steel floors.", 'Everybody wants someone to snuggle with. Even giant armored scaly draconic beasts of destruction.', 'So you say you want an evolution. Well, you know. We all want to change the board.', "It's adorable! AND OH MY GOODNESS WHY IS IT EATING MY FACE", "Kazakus has a squad of imps bottling Felfire round-the-clock and he *still* can't keep up with demand.", 'No Job is too big.  No fee is too big.']
race (object, 7 distinct, 88.4% missing): ['BEAST', 'MECHANICAL', 'DEMON', 'DRAGON', 'MURLOC', 'PIRATE', 'TOTEM']
how_to_earn (object, 29 distinct, 91.3% missing): ['Unlocked at Level 1.', 'Unlocked in the Hall of Explorers, in the League of Explorers adventure.', 'Unlocked in the Parlor, in One Night in Karazhan.', 'Unlocked in the Spire, in One Night in Karazhan.', 'Unlocked in The Ruined City, in the League of Explorers adventure.', 'Unlocked in the Menagerie, in One Night in Karazhan.', 'Unlocked in the Opera, in One Night in Karazhan.', 'Unlocked in the Temple of Orsis, in the League of Explorers adventure.', 'Unlocked in Uldaman, in the League of Explorers adventure.', 'Unlocked at Level 4.']
how_to_earn_golden (object, 80 distinct, 89.7% missing): ['Crafting unlocked in the Hall of Explorers, in the League of Explorers adventure.', 'Crafting unlocked in the Spire, in One Night in Karazhan.', 'Crafting unlocked in the Parlor, in One Night in Karazhan.', 'Crafting unlocked in The Ruined City, in the League of Explorers adventure.', 'Crafting unlocked in the Menagerie, in One Night in Karazhan.', 'Crafting unlocked in the Opera, in One Night in Karazhan.', 'Crafting unlocked in Uldaman, in the League of Explorers adventure.', 'Crafting unlocked in the Temple of Orsis, in the League of Explorers adventure.', 'Unlocked at Level 47.', 'Unlocked at Level 40.']
targeting_arrow_text (object, 41 distinct, 98.0% missing): ['Deal 1 damage.', '<b>Silence</b> a minion.', 'Give +2 Attack.', 'Deal 3 damage.', 'Deal 4 damage.', 'Restore 2 Health.', 'Give +1/+1.', 'Deal 6 damage.', 'Deal 2 damage.', 'Return a minion to your hand.']
faction (object, 2 distinct, 97.7% missing): ['ALLIANCE', 'HORDE']
durability (float64, 7 distinct, 97.7% missing): ['2.0', '3.0', '4.0', '5.0', '8.0', '6.0', '1.0']
"""

import os

import pandas as pd

from multabench.datasets.all_datasets import KaggleDatasetID
from multabench.datasets.downloading import download_dataset
from multabench.benchmark.utils.curation import save_dataset, task_type_from_name


DATASET_ID = "MUL_TEXT_HEARTHSTONE_CARDS"
SLUG_BASE = "multabench-full-hearthstone-cards"
KAGGLE_SOURCE = "https://www.kaggle.com/jeradrose/hearthstone-cards"


def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dataset = download_dataset(KaggleDatasetID.MUL_TEXT_SOCIAL_HEARTHSTONE_CARD_GAME_WARCRAFT)
    df = pd.concat([dataset.x, dataset.y], axis=1)
    save_dataset(df=df, output_dir=output_dir, target_col=dataset.y.name, dataset_id=DATASET_ID,
                 slug=slug, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)


if __name__ == "__main__":
    from multabench.benchmark.utils.curation import parse_curation_args
    args = parse_curation_args(SLUG_BASE, description="Curate MUL_TEXT_HEARTHSTONE_CARDS for MulTaBench-Full")
    curate(output_dir=args.output_dir, slug=args.slug)
