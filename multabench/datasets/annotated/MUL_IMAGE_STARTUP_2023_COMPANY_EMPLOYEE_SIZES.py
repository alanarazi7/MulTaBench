import os
from os.path import exists, join
from typing import Optional

from pandas import DataFrame

from multabench.datasets.utils import load_csv
from multabench.datasets.curation_objects import CuratedTarget, CuratedFeature
from multabench.datasets.objects import SupervisedTask, FeatureType


'''
Dataset Name: chickooo/top-tech-startups-hiring-2023/
====
Examples: 3000
====
URL: https://www.kaggle.com/chickooo/top-tech-startups-hiring-2023
====
Description: 
Top Tech Startups Hiring 2023
A dataset on top tech startups and their hiring information.

About Dataset
This dataset contains data on top tech startups and their hiring information. Some of the listed features are company name, description of the company, company's website URL, industries in which the company operates, hiring activity of the company, job vacancies etc. Please refer to data_dictionary.txt for a complete list of features.

This dataset also has images of the company logo. These images are present in the ./images folder and are named with respect to their company id. The maximum dimensions of each image are 200 x 200 pixels, while some images have rectangular shapes having value below 200 pixels.

The dataset is provided in 2 formats i.e., json and csv. As some features have array and key-value pair as their datatypes, the suitable file format was json. But the scientific community is more inclined towards tabular format hence; csv file format is also provided. Note while transforming data to csv format, array and key-value datatypes were slightly modified. Please refer to the schema files to get a feeling for the structure of these datatypes.

A quick note about missing values. This dataset contains some missing values, they are represented using empty strings "". When working with the data, please take appropriate measures.

At last, most of the irregularities from the dataset were handled, and if you find any, please report it, and it will be fixed in the next version.



Note:
The dataset is for educational use only. The data owners are the respective companies and the websites providing the data.
That being said, the jobs listed in the dataset are real, and if interested, you can apply to them from the respective company's website.
====
Target Variable: employees (object, 7 distinct): ['11-50', '51-200', '1-10', '201-500', '501-1000', '1001-5000', '5000+']
====
Features:

company_name (object, 2941 distinct): ['Loop', 'Arvato Systems Malaysia', 'Cinemaloop', 'Musixmatch', 'Rev', 'Learnyst', 'nOps', 'Scratch', 'Bolt', 'OverHaul']
headline (object, 2940 distinct, 0.5% missing): ['Zero to One Product Innovation for Corporate Incubation & Early stage startups', 'OKR & Strategy Execution Software', 'An Intelligent Driver and Fleet Safety Platform', 'We help manufacturing teams drive efficiency by making knowledge processes collaborative', 'Enabling access to global financial products', 'Modernising small businesses across the world', 'We are an IT service solutions tech resourcing hub that lives and breathes digital technology', 'Providing better banking technology to over 11,500 financial institutions', 'Patient health monitoring devices powered by the cloud', 'Infrastructure for on-chain communities']
about (object, 2916 distinct, 1.4% missing): ['Our VisionTo Build an AI Nervous System, that comprehends human behavior to augment the decision-making capabilities, thereby empowering the Future Mobility Ecosystem. Our MissiondrivebuddyAI aims to accelerate the evolution in the future of mobility through the transformation of the Fleet Management & Insurance Ecosystem. We are on a mission to enable commercial fleets with an AI-powered intelligent driver & fleet safety platform, ensuring drivers’ safety, reducing high-risk-loss-making events, and improving efficiency.', 'Zolve’s mission is simple: we want to make financial products accessible to everyone. Zolve offers the ambitious a plethora of products to simplify banking in the US, such as: Zolve is expanding its horizons, and we’re excited to open our doors to everyone expanding theirs.We believe: that if space tourism is real, so is reaching for the stars.', "The biggest waste in manufacturing today is not on the shop floor. It's in the meetings, emails, and data transfer that overwhelm knowledge processes (eg quality assurance, vendor management, engineering, R&D, and sales). That's because traditional enterprise tools split collaboration and processes into disparate systems, resulting in up to 85% of time and effort being wasted. Unifize solves this problem by bringing collaboration and process into one place. By creating a conversation for every process record, Unifize measurably reduces waste in knowledge processes by up to 95%.", "Good for the planet, Good for humans. Helping lifestyle changemakers to thrive, inspire and build a conducive environment for people. For professionals, Influencers, entrepreneurs, organizations, and communities providing services, products, or just free content. Empower changemakers with technology, data, intelligent workflows, and tools to build integrative & collaborative experiences and get constant help from us to impact people's way of living.", 'Aptos is building a Layer 1 blockchain designed with an emphasis on absolute safety, extensible scalability, and credible neutrality — values that we know firsthand and viscerally understand. Now, on the fourth iteration of the consensus protocol, we are confident that Aptos is capable of bringing affordability, decentralization, and speed to the daily lives of billions of internet users. It is the lowest latency, optimistically-responsive BFT protocol available, and it features a robust, on-chain reputation system and novel methods for parallel execution — a key enabler of speed at scale. We will rapidly deploy many key innovations in performance, functionality and improvements to the overall user experience and leverage our ability to do major upgrades seamlessly in the process.', "Most organizations don't have the resources to focus on reducing cloud spend. nOps is your ML-powered FinOps team. nOps reduces cloud waste, helps you run workloads on spot instances, automatically manages reservations, and helps you to optimize your containers. Everything is automated and data-driven.", 'Every day, entrepreneurs and organizations have good ideas for digital products. Digital products that solve a problem or seize a new opportunity in the market. It is these entrepreneurs and organizations that Your Software Supplier is supporting. Your Software Supplier helps them to realize their idea for a digital product by matching the right software supplier. We do this matchmaking between software customers and software suppliers via our online platform and via our offline matchmaking services. Our online platform offers the possibility to find the right software supplier via various search functions and the supplier map. In addition, software customers can share their project with us, after which we link the right software supplier to the project. Besides, our team of account executives is available worldwide to help software suppliers and software clients. Are you looking for new customers? Or are you looking for the right supplier? Take advantage of our high-end services. At Your Software Supplier we celebrate new relations between software clients and software suppliers.', "In 2018, we started vivenu with a bold mission: to transform the global event ticketing industry for good. Hundreds of thousands of event organizers around the globe rely on solutions that haven’t kept up with today's needs and expectations. Platforms that were built and got stuck in the 90s made the jobs of ticket managers worldwide an ever-lasting misery. We finally put this to an end. Now and forever. Our API-first ticketing platform breaks limitations and unlocks huge potential: letting leading organizers manage, market, and analyze ticket sales effortlessly with our powerful unified solution. Leading tech VCs, exceptional entrepreneurs, and industry experts such as the San Francisco 49ers invested more than $65 million in vivenu to back our extraordinary growth. We are not an ordinary company and we are not looking for people who want just another job. Our team consists of truly driven individuals, working together to achieve the unimaginable. Ready for personal growth? Join us to be part of Germany's next big B2B SaaS company!", "Hitwicket is here to provide an opportunity that brings out your creativity and makes work as fun as playing a game. The Global Gaming industry is worth $206 Billion, with Mobile Gaming itself accounting for more than $100 Billion! Mobile gaming is one of the few sectors that has continued the growth through the pandemic. Hitwicket is a Series A funded Technology startup based in Hyderabad and co-founded by VIT alumni. We are backed by Prime Venture Partners, one of India’s oldest and most successful venture funds - Hitwicket Superstars won the First prize in Prime Minister’s AatmaNirbhar Bharat App Innovation Challenge, a nation-wide contest to identify the top homegrown startups who are building for the Global market; Made in India, for India & the World! Join us in this EPIC journey and make memories for a lifetime! With the phenomenal success of our Cricket Game, we are now entering into the world of Football, NFTs & Blockchain gaming! We are assembling a team to join us on our mission to make something as massive as PUBG or Clash of Clans from India. Our work culture is driven by speed, innovation and passion, not by hierarchy. Our work philosophy is centered around building a company like a 'Sports Team' where each member has an important role to play for the success of the company as a whole.It doesn't matter if you are not a cricket fan, or a gamer, what matters to us are your Problem solving skills and your Creativity. To know more about Hitwicket, please visit our website:", 'OpenSea is the first and largest marketplace for digital collectibles, which include gaming items, digital art, and other virtual goods backed by a blockchain. On OpenSea, anyone can buy or sell these items using smart contracts.']
tags (object, 222 distinct): ['Actively Hiring', 'Actively Hiring, Growing fast', 'Actively Hiring, Recently funded', 'Recently funded', 'Actively Hiring, Same investor as Airbnb', 'Actively Hiring, Highly rated, Work / Life Balance, Strong Leadership', 'Same investor as Airbnb', 'Same investor as Meta', 'Actively Hiring, Same investor as Meta', 'Same investor as PayPal']
locations (object, 1337 distinct, 0.2% missing): ['San Francisco', 'New York City', 'Bengaluru', 'Remote', 'London', 'Los Angeles', 'San Francisco, Remote', 'Mumbai', 'Toronto', 'Boston']
industries (object, 2731 distinct, 0.2% missing): ['SaaS', 'Finance Technology', 'Blockchain / Cryptocurrency', 'Enterprise Software', 'Healthcare', 'Fin Tech', 'Global', 'B2B · SaaS · Mobile · Artificial Intelligence / Machine Learning', 'Financial Services', 'Finance Technology, Blockchain / Cryptocurrency']
jobs (object, 1606 distinct, 0.0% missing): ['(Engineering: 1)', '(Engineering: 2)', '(Sales: 1)', '(Engineering: 3)', '(Marketing: 1)', '(Engineering: 1), (Sales: 1)', '(Engineering: 4)', '(Engineering: 1), (Marketing: 1)', '(Sales: 2)', '(Designer: 1), (Engineering: 1)']
company_Logo (object, 2998 distinct, 0.1% missing): ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg', '6.jpg', '7.jpg', '8.jpg', '9.jpg', '10.jpg']
'''

IMAGE_FEATURE_NAME = "company_Logo"


def load_df(dir_path: str) -> DataFrame:
    df = load_csv(dir_path, "csv_data.csv")
    df[IMAGE_FEATURE_NAME] = df['id'].apply(lambda i: get_img(i, dir_path))
    return df

def get_img(img: str, dir_path: str) -> Optional[str]:
    img_folder = join(dir_path, IMAGE_FOLDER)
    img_filename = f"{img}.jpg"
    if not exists(join(img_folder, img_filename)):
        return None
    return img_filename


CONTEXT = ""
TARGET = CuratedTarget(raw_name='employees', task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = ['id', 'website']
FEATURES = [CuratedFeature(raw_name=IMAGE_FEATURE_NAME, feat_type=FeatureType.IMAGE)]
IMAGE_FOLDER = "images/images"
LOADING_FUNC = load_df
