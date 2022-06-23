from gen_data import get_latest_data
from train import get_model, checkpoint_filepath

symbols_list = ['SPY','QQQ','IWM','GLD','TSLA','NVDA','NFLX','AMC']
x = get_latest_data(symbols=symbols_list)
print(x.shape)

model = get_model()
model.load_weights(checkpoint_filepath)
y = model.predict(x)

for ticker,pred in zip(symbols_list,y):
    print(ticker,pred)

