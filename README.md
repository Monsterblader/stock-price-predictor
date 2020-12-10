# Predicting Stock Prices with Machine Learning

Using tools that I learned at Metis data-science bootcamp.

## Goal of this project

Use machine learning strategies, in this case, Long Short-Term Memory, to analyze closing stock prices from yahoo! finance then predict the stock's closing prices.

## Sources

The data is acquired from [yahoo! finance](https://finance.yahoo.com).  The trading indicators is taken from [FinTA](https://github.com/peerchemist/finta)

## Significant files

* Project 5 presentation.pdf - screenshots of the presentation.
* analysis/
  + pred.ipynb - the prediction generator.
* HTML/ - the presentation files.

## Tools

TensorFlow and Keras

 * Long Short-Term Memory
 * Technical indicators from FinTA

## Takeaways

* I did not use the indicators as intended.  They are mostly trinary, i.e. buy, sell, or hold.  While their values can be calculated daily, they should be ignored unless they are indicating buy or sell.
* As the number of epochs increased, the predictions went to zero.  I did not notice this behavior early enough to determine why this was happening.

## Future work

* Recreate the model using Facebook Prophet.
* Redesign the LSTM model to use the indicators as -1, 0, 1.
* As target price is not as necessary as direction and magnitude of move, a categorical model might be better.