from os.path import join

from pandas import DataFrame, read_csv

from multabench.datasets.curation_objects import CuratedTarget
from multabench.datasets.objects import SupervisedTask

'''
Dataset Name: dhanushbommavaram/laptop-dataset/
====
Examples: 984
====
URL: https://www.kaggle.com/dhanushbommavaram/laptop-dataset
====
Lennart:
This dataset contains laptop listings with specifications, textual descriptions, and target prices. Inputs include categorical
traits (e.g., Processor Brand), numerical specs (e.g., RAM, Screen Size), and verbose text (e.g., Sales Package, Additional
Features). Rather than filtering, all features are retained to simulate real-world noise and redundancy. It serves as a
robust stress test for tabular foundation models under messy conditions.
====
Description:
The dataset is the laptop listings on Flipkart, an ecommerce website in India. This is an unprocessed dataset. It contains 984 unique entries. This unprocessed dataset contains 96 features (Ex: Laptop name, link to buy, price in Indian rupees, processor, GPU, etc.). This dataset was created by scraping data from the web (dated June).
====
Target Variable: Price (float64, 433 distinct): ['40990.0', '38990.0', '54990.0', '64990.0', '58990.0', '42990.0', '79990.0', '65990.0', '49990.0', '34990.0']
====
Features:

name (object, 943 distinct): ['DELL Ryzen 5 Hexa Core 5600H - (8 GB/512 GB SSD/Windows 11 Home/4 GB Graphics/NVIDIA GeForce RTX 3050/120 Hz) G15-5515 Gaming Laptop  (15.6 inch, Grey, 2.57 kg, With MS Office)', ...]
user rating (float64, 28 distinct, 29.9% missing): ['4.3', '4.4', '4.5', '4.2', '4.1', '4.0', '4.6', '4.7', '3.8', '3.9']
Sales Package (object, 106 distinct): ['Laptop, Power Adaptor, User Guide, Warranty Documents', ...]
Model Number (object, 712 distinct): ['Inspiron 3511', 'Vostro 3400', ...]
Part Number (object, 923 distinct): ['82ND007VIN', '34W78PA#ACJ', ...]
Model Name (object, 469 distinct, 27.9% missing): ['Inspiron 3511', 'Inspiron 5410', ...]
Series (object, 257 distinct, 20.0% missing): ['Inspiron', 'Vostro', 'Pavilion', 'IdeaPad 3', ...]
Color (object, 132 distinct): ['Black', 'Transparent Silver', ...]
Type (object, 9 distinct): ['Thin and Light Laptop', 'Gaming Laptop', 'Laptop', '2 in 1 Laptop', 'Notebook', 'Chromebook', '2 in 1 Gaming Laptop', 'Business Laptop', 'Creator Laptop']
Suitable For (object, 52 distinct): ['Processing & Multitasking', 'Gaming', 'Everyday Use', ...]
Power Supply (object, 115 distinct, 44.3% missing): ['65 W AC Adapter', '45 W AC Adapter', ...]
Battery Cell (object, 31 distinct, 30.9% missing): ['3 cell', '3', '3 Cell', '4 cell', ...]
MS Office Provided (object, 2 distinct): ['Yes', 'No']
Dedicated Graphic Memory Type (object, 5 distinct, 65.7% missing): ['GDDR6', 'GDDR5', 'DDR4', 'DDR5', 'GDDR3']
Dedicated Graphic Memory Capacity (object, 9 distinct, 68.5% missing): ['4 GB', '2 GB', '6 GB', '8 GB', ...]
Processor Brand (object, 5 distinct): ['Intel', 'AMD', 'Apple', 'Qualcomm', 'MediaTek']
Processor Name (object, 30 distinct): ['Core i5', 'Core i3', 'Core i7', 'Ryzen 5 Hexa Core', ...]
Processor Generation (object, 9 distinct, 29.9% missing): ['11th Gen', '10th Gen', '12th Gen', ...]
SSD (object, 2 distinct): ['Yes', 'No']
SSD Capacity (object, 5 distinct, 11.0% missing): ['512 GB', '256 GB', '1 TB', '128 GB', '2 TB']
RAM (object, 4 distinct): ['8 GB', '16 GB', '4 GB', '32 GB']
RAM Type (object, 8 distinct): ['DDR4', 'LPDDR4X', 'DDR5', 'LPDDR4', 'LPDDR5', 'LPDDR3', 'Unified Memory', 'DDR3']
Processor Variant (object, 132 distinct, 12.5% missing): ['1135G7', '1115G4', ...]
Clock Speed (object, 350 distinct, 12.5% missing): ['2.4 GHz with Turbo Boost Upto 4.2 GHz', ...]
Expandable Memory (object, 36 distinct, 61.1% missing): ['16', 'Upto 16 GB', '32', ...]
Cache (object, 42 distinct, 22.7% missing): ['8', '8 MB', '4', '6', ...]
Graphic Processor (object, 151 distinct, 2.0% missing): ['Intel Integrated UHD', 'Intel Integrated Iris Xe', 'NVIDIA GeForce GTX 1650', ...]
Number of Cores (float64, 8 distinct, 26.4% missing): ['4.0', '2.0', '8.0', '6.0', '14.0', '12.0', '1.0', '10.0']
OS Architecture (object, 2 distinct, 16.1% missing): ['64 bit', '32 bit']
Operating System (object, 10 distinct): ['Windows 10 Home', 'Windows 11 Home', 'Windows 10', 'Chrome', 'Windows 10 Pro', 'Mac OS Big Sur', 'DOS', 'Windows 11 Pro', 'Mac OS Monterey', 'Ubuntu']
Supported Operating System (object, 34 distinct, 50.3% missing): ['Windows 11 Home', 'Windows 11', ...]
Mic In (object, 2 distinct, 18.0% missing): ['Yes', 'No']
USB Port (object, 401 distinct, 10.6% missing): ['1 x USB 3.2 Gen 1 Type-A, 1 x USB 3.2 Gen 1 Type-C, 2 x USB 2.0 Type-A', ...]
HDMI Port (object, 65 distinct, 18.3% missing): ['1 x HDMI 1.4', '1 x HDMI Port', ...]
Touchscreen (object, 2 distinct): ['No', 'Yes']
Screen Size (object, 38 distinct): ['39.62 cm (15.6 inch)', '35.56 cm (14 inch)', ...]
Screen Resolution (object, 72 distinct): ['1920 x 1080 Pixel', '1920 x 1080 Pixels', ...]
Screen Type (object, 518 distinct, 10.3% missing): ['Full HD WVA AG Narrow Border', ...]
Speakers (object, 76 distinct, 13.4% missing): ['Built-in Dual Speakers', ...]
Internal Mic (object, 69 distinct, 14.3% missing): ['Yes', 'Built-in Microphone', ...]
Sound Properties (object, 203 distinct, 29.1% missing): ['Waves Maxx Audio Pro', ...]
Wireless LAN (object, 170 distinct, 11.8% missing): ['802.11ac', ...]
Bluetooth (object, 26 distinct, 0.4% missing): ['v5.0', 'v5.1', 'v5.2', 'v4.1', 'YES', 'v4.2', 'yes', 'Yes', '4.2', '4.1']
Dimensions (object, 380 distinct, 12.3% missing): ['360.2 x 234.9 x 19.9 mm', ...]
Weight (object, 207 distinct, 11.3% missing): ['1.80 kg', '1.8 kg', ...]
Disk Drive (object, 3 distinct): ['Not Available', 'CD/DVD writer', 'CD/DVD reader']
Finger Print Sensor (object, 2 distinct, 27.4% missing): ['No', 'Yes']
Keyboard (object, 189 distinct, 10.3% missing): ['Backlit Chiclet Keyboard', ...]
Backlit Keyboard (object, 2 distinct, 17.1% missing): ['Yes', 'No']
Additional Features (object, 296 distinct, 41.4% missing): ['Li-ion Battery', ...]
Warranty Summary (object, 81 distinct): ['1 Year Onsite Warranty', ...]
Warranty Service Type (object, 25 distinct): ['Onsite', 'Carry-In Warranty', ...]
Covered in Warranty (object, 36 distinct): ['Manufacturing Defects', ...]
Not Covered in Warranty (object, 42 distinct, 0.3% missing): ['Physical Damage', ...]
Domestic Warranty (object, 6 distinct, 17.3% missing): ['1 Year', '2 Year', '3 Year', '12 Months', '36 Months', '24 Months']
Ethernet (object, 30 distinct, 68.5% missing): ['Available', 'Yes', ...]
Web Camera (object, 140 distinct, 14.3% missing): ['HD', 'HD Webcam', ...]
Pointer Device (object, 46 distinct, 41.4% missing): ['Touchpad', ...]
Included Software (object, 171 distinct, 38.6% missing): ['Microsoft Office Home and Student 2019', ...]
Battery Backup (object, 152 distinct, 38.2% missing): ['Upto 10 Hours', ...]
Chipset (object, 29 distinct, 86.0% missing): ['Intel SoC Platform', ...]
Memory Slots (object, 17 distinct, 69.5% missing): ['2', '2 Slots', ...]
RAM Frequency (object, 19 distinct, 60.1% missing): ['3200 MHz', ...]
RJ45 (object, 2 distinct, 57.5% missing): ['Yes', 'No']
Sound Chip (object, 33 distinct, 88.7% missing): ['High Definition (HD) Audio', ...]
Brightness (object, 4 distinct, 98.8% missing): ['300 Nits', '250 Nits', '400 Nits', '220 Nits']
Laptop Bag (object, 2 distinct, 80.8% missing): ['No', 'Yes']
Other Accessories (object, 18 distinct, 96.0% missing): ['Power Adaptor', 'Sleeve', ...]
International Warranty (object, 6 distinct, 85.7% missing): ['1 Year', '2 Year', '12 Months', '0 Months', '36 Months', '1']
Wireless WAN (object, 30 distinct, 93.7% missing): ['Yes', ...]
Recovery Options (object, 8 distinct, 96.7% missing): ['Yes', 'One Touch Recovery', ...]
RPM (float64, 6 distinct, 90.5% missing): ['5400.0', '7200.0', '4.0', '2666.0', '3200.0', '8.0']
Hardware Interface (object, 47 distinct, 45.1% missing): ['PCIe NVMe', ...]
Face Recognition (object, 2 distinct, 96.5% missing): ['No', 'Yes']
System Architecture (object, 11 distinct, 42.5% missing): ['64', '64 bit', ...]
Refresh Rate (object, 11 distinct, 84.0% missing): ['144 Hz', '60 Hz', '120 Hz', ...]
Antivirus (object, 15 distinct, 86.7% missing): ['McAfee', ...]
Multi Card Slot (object, 72 distinct, 48.6% missing): ['1 x Micro SD Card Reader', ...]
Lock Port (object, 17 distinct, 77.2% missing): ['Kensington Lock Port', ...]
NFC Support (object, 1 distinct, 95.8% missing): ['No']
HDD Capacity (object, 2 distinct, 81.5% missing): ['1 TB', '512 GB']
Stylus Included (object, 2 distinct, 94.8% missing): ['No', 'Yes']
TGP (object, 12 distinct, 97.1% missing): ['35 W', '60 W', '50 W', ...]
VGA Port (object, 2 distinct, 90.8% missing): ['No', 'Yes']
Color Gamut (object, 3 distinct, 99.6% missing): ['100% DCI-P3 color gamut, PANTONE Validated', '75%', '75']
Security Chip (object, 19 distinct, 85.6% missing): ['Firmware TPM 2.0', 'TPM 2.0', ...]
Dock Port (object, 1 distinct, 96.6% missing): ['No']
Firewire Port (object, 1 distinct, 96.5% missing): ['No']
RJ11 (object, 1 distinct, 95.8% missing): ['No']
EMMC Storage Capacity (object, 4 distinct, 97.6% missing): ['64 GB', '128 GB', '4 GB', '32 GB']
'''

def load_df(dir_path: str) -> DataFrame:
    df_path = join(dir_path, "complete laptop data0.csv")
    df = read_csv(df_path, encoding="utf-8", encoding_errors="replace")
    df['Price'] = df['Price'].apply(clean_rupi)
    return df


def clean_rupi(price: str) -> float:
    # Rupi looks like: ?2,34,990
    assert price.startswith("?")
    price = price.strip("?").replace(",", "")
    return float(price)

CONTEXT = ""
COLS_TO_DROP = [
    # Index
    "Unnamed: 0",
    # Non-Image URL
    "link",
    # Constant / Highly missing
    "Cloud Storage", "Optane Memory", "Read/Write Speed", "S-video", "Inbuilt 4G LTE",
    ]
TARGET = CuratedTarget(raw_name="Price", task_type=SupervisedTask.REGRESSION)
LOADING_FUNC = load_df
