
```

docker run -it -u $(id -u):$(id -g) -w $PWD -v /mnt:/mnt  pangyuteng/ml:latest bash


data downloaded from 
https://transparentcalifornia.com
https://transparentcalifornia.com/download/salaries/university-of-california

rm *.json *.txt

python plot_salary_progression.py


# work-in-progress
python plot_salary_progression_word2vec.py

```
