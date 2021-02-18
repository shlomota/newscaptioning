# Label Generation + POC
* **Shlomo** BM25 – find python package and test it out on an article. Is the chosen paragraph good?
  * Look into other ranking scores – rankings from paper?   BLEU-1 BLEU-2 BLEU-3 BLEU-4 ROUGE METEOR CIDEr
  * Test and compare different rankings 
* **Maor** Feed single paragraph to caption model generator
  * o	“Fake” article with one paragraph
* **Ron** Retrieve score from model evaluation (bubble 3 – Similarity(generated caption, true caption) )
* Compare scores:
similarity(generated caption, true caption) vs. similarity(paragraph, generated caption)
High correlation? Yay.

## POC flowchart
![tasks](https://github.com/shlomota/newscaptioning/blob/master/Managerial/label_generation_flowchart.jpg?raw=true)
