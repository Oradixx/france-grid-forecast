{% test dbt_utils_free_unique_combination(model, columns) %}
-- Same as dbt_utils.unique_combination_of_columns, without the package dependency
select {{ columns | join(', ') }}, count(*) as n
from {{ model }}
group by {{ columns | join(', ') }}
having count(*) > 1
{% endtest %}
