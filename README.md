Right now, a fine-tuned ROBERTa scores 82.89% accuracy in just 5 epochs. Can we achieve 90?

### Plan of attack:

- [ ] See the most commonly used disaster words (not counting stopwords) and non-disaster.
- [ ] Do the BoW / TF-IDF appraoch where we would select the most occuring disaster-words and weigh this more in attention layer?
- [x] ROBERTa method (done) - 82.89% acc with reduced LR.
- [x] preprocessing
- [ ] ensemble method - SEE Why this works 

> "The reason that model averaging works is that different models will usually not make all the same errors on the test set."

https://machinelearningmastery.com/ensemble-methods-for-deep-learning-neural-networks/

- [ ] evaluate on other metrics too like F1, Precision, Recall, AUROC

