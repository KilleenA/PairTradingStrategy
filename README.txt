Module to back test pairs trading strategy and perform index regression analysis.

N.B. As well as the modules listed in requirements.txt, the user must also have Java installed.

Executive Summary

-Run PairTradingGUI to bring up a GUI to set up trade pairing parameters which can be
run by pressing 'Trade' or optimised by pressing 'Optimise'. 

-Correlation analysis between pairs of stocks can be done by running EvaluateTradePairs
(this takes a couple of minutes) or by setting 'eval_pairs' to 'True' when calling 
'TradePreprocessing' in main in PairTradingGUI.

-Run IndexRegressionGUI to bring up a GUI to select target index to regress and choose
 a list of up to 10 stocks on which to perform regression

-Stock and index data can be updated by running DataLoader or by setting 'update_data'
to 'True' in main of PairTradingGUI or IndexRegressionGUI 



PairTradingGUI

Pair trade back testing and parameter optimisation can be performed by running 'PairTradingGUI'.
By default this doesn't check if the downloaded stock data needs updating and doens't run a 
correlation analysis of different pair combinations. However these can be done by setting 'update_data'
and 'eval_pairs' to true when calling 'TradePreprocessing' in main.

If eval_pairs is set to true this performs a correlation analysis of all possible pair combinations by 
calling EvaluateTradePairs in the file EvaluateTradePairs. Constraints on desired correlation metrics
(such as range of mean reversion times, minimum acceptable correlation strength, maximum acceptable hurst
exponent etc.) can be set in the file EvaluateTradePairs. As this examines millions of combinations
of stock pairs it can be quite slow, so it is suggested to use tight contraints on acceptable correlation
metrics. 

EvaluateTradePairs will print out up to ten highly correlated pairs the fit the required constraints for the
user to use for their trading strategy. To see all pairs that fit the required constraints, the user should
run EvaluateTradePairs on its own. This will output all pairs in a dataframe called 'pair_metrics'.

If update_data is set to true then any new price data that has become available since the last download
will be downloaded. N.B. Due to delisting of / changes to Russell 2000 components, some of the acquired 
tickers will fail to download data.

If desired, by setting 'redownload' to true when calling 'TradePreprocessing'
the user can reset the downloaded data and redownload all stock data for the last ten years, although this
may take a few minutes. Alternatively, the latest stock data can be acquired and saved on its own by running
DataLoader. 

Parameter optimisation is done using a DyCors scheme, a derivative free surrogate modelling 
optimisation scheme. Please see https://www.tandfonline.com/doi/abs/10.1080/0305215X.2012.687731 
for details of methodology and implementation.The bounds for the parameters in the optimisation are 
the same as those on the sliders in the GUI. As the Kalman method is faster than the OLS method, 
it is recommended to use this method when optimising parameters.

If the strategy loses more than the initial equity a bankruptcy warning will be printed to the screen but 
the back test will continue, meaning any error messages may be printed to the screen. This is most likely to
happen at the start of a parameter optimisation.

IndexRegressionGUI

By default IndexRegressionGUI also doesn't check if the downloaded stock data needs updating. 
This can again be changed by setting 'update_data' to true in main. 


