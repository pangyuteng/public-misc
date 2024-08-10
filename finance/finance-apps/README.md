
#### undeployed!

```
source /mnt/scratch/venv/agw-com/bin/activate
aws configure
zappa undeploy prod

above failed, manually deleting, route 53 dns, CloudFormation, Lambda, API Gateway (APIs + Custom domain names), s3, Certificate manager.
Amazon EventBridge->Rules
```
```

python3.8 -m venv /mnt/scratch/venv/finance
source /mnt/scratch/venv/finance/bin/activate
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