"""
Dataset Name: REG_TEXT_ROTTEN_TOMATOES
====
Examples: 7158
====
Target Variable: RatingValue (float64, 82 distinct): ['6.7', '6.4', '7.2', '6.5', '7.1', '6.8', '6.6', '7.0', '6.2', '7.3']
====
Features:

Name (object, 6975 distinct): ['Treasure Island', 'The Prisoner of Zenda', 'Jack and the Beanstalk', 'The Last of the Mohicans', 'Little Women', 'Black Beauty', 'The Raven', 'Honeymoon', 'King Kong', 'Fair Game']
Year (int64, 103 distinct): ['2014', '2015', '2013', '2012', '2011', '2008', '2010', '2009', '2006', '2007']
Release Date (object, 4710 distinct, 1.0% missing): ['14 August 2015 (USA)', '2010 (USA)', '25 September 2015 (USA)', '2015 (USA)', '17 July 2015 (USA)', '18 September 2015 (USA)', '17 October 2014 (USA)', '15 May 2015 (USA)', '21 August 2015 (USA)', '7 August 2015 (USA)']
Director (object, 3586 distinct): ['Woody Allen', 'Clint Eastwood', 'Steven Spielberg', 'Martin Scorsese', 'Billy Wilder', 'Alfred Hitchcock', 'Oliver Stone', 'Michael Curtiz', 'Steven Soderbergh', 'Ridley Scott']
Creator (object, 5982 distinct, 2.8% missing): ['Woody Allen', 'John Hughes', 'John Waters', 'Joel Coen,Ethan Coen', 'M. Night Shyamalan', 'Kevin Smith', 'Hal Hartley', 'Jason Friedberg,Aaron Seltzer', 'Michael Moore', 'Frances Goodrich,Albert Hackett']
Actors (object, 6999 distinct, 0.8% missing): ['Bing Crosby,Bob Hope,Dorothy Lamour', 'Daniel Radcliffe,Emma Watson,Rupert Grint', 'Groucho Marx,Chico Marx,Harpo Marx', 'Winston Hibler', 'William Shatner,Leonard Nimoy,DeForest Kelley', 'Groucho Marx,Harpo Marx,Chico Marx', 'Sylvester Stallone,Talia Shire,Burt Young', 'Kristen Stewart,Robert Pattinson,Taylor Lautner', 'Divine,David Lochary,Mary Vivian Pearce', 'Clint Eastwood,Sondra Locke,Geoffrey Lewis']
Cast (object, 7092 distinct, 0.8% missing): ['Winston Hibler', "Ian McKellen,Martin Freeman,Richard Armitage,Ken Stott,Graham McTavish,William Kircher,James Nesbitt,Stephen Hunter,Dean O'Gorman,Aidan Turner,John Callen,Peter Hambleton,Jed Brophy,Mark Hadlow,Adam Brown", 'Kurt Russell,Zoë Bell,Rosario Dawson,Vanessa Ferlito,Sydney Tamiia Poitier,Tracie Thoms,Rose McGowan,Jordan Ladd,Mary Elizabeth Winstead,Quentin Tarantino,Marcy Harriell,Eli Roth,Omar Doom,Michael Bacall,Monica Staggs', 'Paul Sanchez,Lari White,Leonid Citer,David Allen Brooks,Yelena Popovic,Valentina Ananina,Semion Sudarikov,Tom Hanks,Peter Von Berg,Dmitri S. Boudrine,François Duhamel,Michael Forest,Viveka Davis,Nick Searcy,Jennifer Choe', "Nastassja Kinski,Malcolm McDowell,John Heard,Annette O'Toole,Ruby Dee,Ed Begley Jr.,Scott Paulin,Frankie Faison,Ron Diamond,Lynn Lowry,John Larroquette,Tessa Richarde,Patricia Perkins,Berry Berenson,Fausto Barajas", 'James Caan,Tuesday Weld,Willie Nelson,James Belushi,Robert Prosky,Tom Signorelli,Dennis Farina,Nick Nickeas,W.R. Brown,Norm Tobin,John Santucci,Gavin MacFadyen,Chuck Adamson,Sam Cirone,Spero Anast', 'Jamie Foxx,Chris Cooper,Jennifer Garner,Jason Bateman,Ashraf Barhom,Ali Suliman,Jeremy Piven,Richard Jenkins,Tim McGraw,Kyle Chandler,Frances Fisher,Danny Huston,Kelly AuCoin,Anna Deavere Smith,Minka Kelly', 'Joey Cramer,Paul Reubens,Veronica Cartwright,Cliff De Young,Sarah Jessica Parker,Albie Whitaker,Matt Adler,Howard Hesseman,Robert Small,Jonathan Sanger,Iris Acker,Richard Liberty,Raymond Forchion,Cynthia Caquelin,Ted Bartsch', 'Ethan Hawke,River Phoenix,Bobby Fite,Bradley Gregg,Georg Olden,Chance Schwass,Amanda Peterson,Danny Nucci,Jason Presson,Dana Ivey,Taliesin Jaffe,James Cromwell,Brooke Bundy,Tricia Bartholome,Eric Luke', 'M.C. Gainey,Paul Soter,Erik Stolhanske,Cloris Leachman,Jürgen Prochnow,Cameron Scher,Owain Yeoman,Tom Tate,Allan Graf,Chris Moss,Bjorn Johnson,Kevin Heffernan,Jay Chandrasekhar,Steve Lemme,Collin Thornton']
Language (object, 500 distinct): ['English', 'English,Spanish', 'English,French', 'English,German', 'English,Italian', 'English,Russian', 'English,Japanese', 'English,Mandarin', 'English,Arabic', 'English,Latin']
Country (object, 393 distinct): ['USA', 'UK,USA', 'USA,UK', 'USA,Germany', 'USA,Canada', 'Canada,USA', 'USA,Australia', 'USA,France', 'Germany,USA', 'France,USA']
Duration (float64, 161 distinct, 1.8% missing): ['90.0', '93.0', '95.0', '91.0', '100.0', '92.0', '96.0', '97.0', '88.0', '94.0']
RatingCount (object, 5902 distinct): ['21', '18', '20', '30', '15', '26', '123', '239', '13', '16']
ReviewCount (object, 5292 distinct, 1.9% missing): ['1 user', '2 user', '1 critic', '3 user', '1 user,1 critic', '4 user,1 critic', '3 user,1 critic', '4 user,3 critic', '4 user', '2 user,1 critic']
Genre (object, 627 distinct, 0.2% missing): ['Drama', 'Comedy', 'Comedy,Romance', 'Comedy,Drama,Romance', 'Horror', 'Documentary', 'Comedy,Drama', 'Drama,Romance', 'Horror,Thriller', 'Action,Crime,Drama']
Filming Locations (object, 3154 distinct, 13.1% missing): ['Los Angeles, California, USA', 'New York City, New York, USA', 'Santa Clarita, California, USA', 'Metro-Goldwyn-Mayer Studios - 10202 W. Washington Blvd., Culver City, California, USA', 'California, USA', 'Warner Brothers Burbank Studios - 4000 Warner Boulevard, Burbank, California, USA', 'Universal Studios - 100 Universal City Plaza, Universal City, California, USA', 'Chicago, Illinois, USA', 'USA', 'Paramount Studios - 5555 Melrose Avenue, Hollywood, Los Angeles, California, USA']
Description (object, 7088 distinct): ['Add a Plot', 'Peaceable Kingdom: The Journey Home explores the powerful struggle of conscience experienced by several people from traditional farming backgrounds who come to question the basic ...', 'Huckleberry Finn, a rambunctious boy adventurer chafing under the bonds of civilization, escapes his humdrum world and his selfish, plotting father by sailing a raft down the Mississippi ...', 'A boy obsessed with 50s Sci-Fi movies about aliens has a recurring dream about a blueprint of some kind, which he draws to his inventor friend. With the help of a third kid, they follow it and build themselves a spaceship. Now what?', "Two brothers travel to Germany for Oktoberfest, only to stumble upon a secret, centuries-old competition described as a 'Fight Club' with beer games.", 'New York police officer Ralph Sarchie investigates a series of crimes. He joins forces with an unconventional priest, schooled in the rites of exorcism, to combat the possessions that are terrorizing their city.', "In New York City, a crime lord's right-hand man is seduced by a woman seeking retribution.", "After his wife is assaulted, a husband enlists the services of a vigilante group to help him settle the score. Then he discovers they want a 'favor' from him in return.", 'A high-profile terrorism case unexpectedly binds together two ex-lovers on the defense team - testing the limits of their loyalties and placing their lives in jeopardy.', 'To protect his brother-in-law from a drug lord, a former smuggler heads to Panama to score millions of dollars in counterfeit bills.']
"""

import os

import pandas as pd

from multabench.datasets.all_datasets import UrlDatasetID
from multabench.datasets.downloading import download_dataset
from multabench.benchmark.utils.curation import save_dataset, task_type_from_name


DATASET_ID = "REG_TEXT_ROTTEN_TOMATOES"
SLUG_BASE = "multabench-rotten-tomatoes"
KAGGLE_SOURCE = "http://pages.cs.wisc.edu/~anhai/data/784_data/movies1/csv_files/rotten_tomatoes.csv"



def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dataset = download_dataset(UrlDatasetID.REG_TEXT_SOCIAL_MOVIES_ROTTEN_TOMATOES)
    df = pd.concat([dataset.x, dataset.y], axis=1)
    save_dataset(df=df, output_dir=output_dir, target_col=dataset.y.name, dataset_id=DATASET_ID,
                 slug=slug, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)
