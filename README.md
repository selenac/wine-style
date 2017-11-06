# Wine Style
##### Selena Chen | Galvanize DSI Capstone Project | Nov. 2017

-------

## Motivation

I'm an individual that enjoys wine. A full body, earthy, and oaky Napa Valley Cabernet Sauvignon paired with cheese or on its own, and I'm content. The frustration in exploring new wines, is often being disappointed or overwhelmed by suggestions from a restaurant sommelier or a wine store clerk.

What if we could base these suggestions off of wines that we already know and love? Enter, Wine Style.

## A Recommendation System & The Data

The recommender is built on item content-based similarity between wines and expert reviews. This is a cold-start recommender as I was unable to gather user specific data.

The data is from Kaggle and features 150k wines scraped from winemag.com. This is the website for Wine Enthusiast magazine.

After scraping additional wines not included in original dataset, cleaning and removing duplicates from the dataset, the result is 118k+ unique wines from all over the world and unique reviews from experts.

The text in these reviews are detailed in describing the flavors, tasting notes, textures, and feelings.

## Approach

The approach I used to make satisfactory recommendations included tokenizing the text, removing common stop words and additional wine specific words (i.e. varietals), and lemmatize with part of speech tagging (i.e. only wanted descriptive nouns and adjectives).

Using the cleaned corpus of wine reviews, I used TFIDF Vectorization to create a matrix on all wines and features (~270 words).

Next, cosine similarity on this TFIDF matrix creates a wine by wine comparison of descriptions and similarities. The recommendation is offered by the highest cosine similarity values for a given wine.

Other approaches that were considered but not included: a) aggregating descriptions by winery and varietal, but many wines over the years would have very different notes; and b) Latent Dirichlet Allocation (LDA) for topic modeling, but the results did not offer satisfactory recommendations. I could judge this by often seeing very different wines (white and red) offered as a recommendation a single wine.]

## Web Application

The recommender product is available on a Flask and Heroku application. The website is: [www.winestyleapp.com](http://www.winestyleapp.com)

The user is able to find recommendations two ways:

* Select a Bottle: Pick a bottle from a random sample of wines from the wine library. Refresh the screen to get a new random sample.
* Describe by Taste: User can enter in a description of the taste and flavors they enjoy in a wine to generate suggestions.

## Validation

Validation for recommenders are a little tricky and the data has wines that readily available to taste. I used a survey to compare the Wine Style recommendations to other website recommendations for a random collection of wines. The respondents were asked to blindly select the recommendations they believed were the strongest.

## Next Steps

I would like to link the recommendations to a wine website with live inventory availability. This would allow for actionable recommendations. I'd also like to add groups to categorize the wine by price, type, etc.
