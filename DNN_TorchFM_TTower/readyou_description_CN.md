## The Process of run the Demo Recommandation System
> 05/01/2025

## I recommand you to use the following steps to run the recommender system.
## it's better to use the Conda env to run the recommender system.

python -m venv rec_python10
python activate rec_python10/scripts/activate

# 1. install the requirment packages
pip install --upgrade -r requirements.txt

# 3. retrain the call back model 
python -m models.recall.train_two_tower --epochs 3 --batch 128

# 4. retraion the re-rank model
python -m models.ranking.train_ranking --epochs 3

# 5. check the recommendation results
python -m service/recommender.py 1001 --top 10

# exit the conda env
deactivate

### Backup code for checkout the database
psql -h postgresql-yannr.alwaysdata.net -p 5432 -U yannr_01 -d yannr_00
Project1234



查看所有表的字段、数据类型、是否可空、默认值


SELECT
    table_name AS 表名,
    column_name AS 字段名,
    data_type AS 数据类型,
    is_nullable AS 是否可空,
    column_default AS 默认值
FROM
    information_schema.columns
WHERE
    table_schema = 'public'
ORDER BY
    table_name, ordinal_position;