```
python3.8 -m venv ~/virtual_env/myvenv
source ~/virtual_env/myvenv/bin/activate
pip install -r requirements.txt

zappa init

-- stage: prod
-- lambda name: zappa-aigonewrong-finance

echo $(openssl rand -hex 4)
append above to s3_bucket in `zappa_settings.json`

zappa deploy prod
zappa update prod


api-gateway, api mappings
finance-apps-prod, prod, `finance/us-market-misc-plots`

```