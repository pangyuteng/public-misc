```
python3.8 -m venv myvenv
source myvenv/bin/activate
pip install -r requirements.txt

zappa init

-- stage: prod
-- lambda name: zappa-aigonewrong-finance

zappa deploy prod
zappa update prod

```