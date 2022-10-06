from transformers import pipeline
classifier = pipeline("sentiment-analysis")
ggh = """My wife had her day pack stolen at Beinglas Farm. The pack was stolen after the host of the farm insisted we put it in the baggage shed. There were no locks on the shed. My wife lost her heart and asthma medication as well as $800 worth of clothing and equipment. We immediately contacted Macs for support and assistance. We received nothing. My wife was able to continue the trip after fellow hikers donated equipment. However she was in constant risk due to her lack of heart and asthma medication and was unable to complete large sections of the trail. The lack of advocacy and support by Macs was unconscionable."""

import spacy
import matplotlib.pyplot as plt
# colors
colors = ['#00F00F', '#0000FF']
# explosion
explode = (0.05, 0.05)
# Load English tokenizer, tagger, parser and NER
nlp = spacy.load("en_core_web_sm")
doc = nlp(ggh)
pos = neg = 0
ps= 0
for entity in doc.sents:
    ff=classifier(str(entity))
    print(entity,ff)
    if ff[0]["score"] > 0.85:
      if ff[0]["label"] == "POSITIVE":
        pos = ff[0]["score"] + pos
        
      else:
        neg = ff[0]["score"] + neg
      ps = ps + 1
    else:
      pass

    pos_per = (pos/ps)/((neg/ps)+(pos/ps)) * 100
    neg_per = (neg/ps)/((pos/ps)+(neg/ps)) * 100
print("positivity is", pos_per)

print("negetivity is", neg_per)

test = [pos_per,neg_per]
X = ["positive","negative"]

plt.pie(test, colors=colors, labels=X,
        autopct='%1.1f%%', pctdistance=0.85,
        explode=explode)
  
# draw circle
centre_circle = plt.Circle((0, 0), 0.60, fc='white')
fig = plt.gcf()
  
# Adding Circle in Pie chart
fig.gca().add_artist(centre_circle)
  
# Adding Title of chart
plt.title('Review Sentiment')
  
# Displaying Chart
plt.show()
