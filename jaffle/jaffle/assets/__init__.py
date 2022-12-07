from dagster import asset
from jaffle.duckpond import SQL
import pandas as pd


@asset
def population() -> SQL:
    df = pd.read_html(
        "https://en.wikipedia.org/wiki/List_of_countries_by_population_(United_Nations)",
    )[0]
    df.columns = [
        "country",
        "continent",
        "subregion",
        "population_2018",
        "population_2019",
        "pop_change",
    ]
    df["pop_change"] = [
        float(str(row).rstrip("%").replace("\u2212", "-")) for row in df["pop_change"]
    ]
    return SQL("select * from $df", df=df)


@asset
def continent_population(population: SQL) -> SQL:
    return SQL(
        "select continent, avg(pop_change) as avg_pop_change from $population group by 1 order by 2 desc",
        population=population,
    )


@asset(required_resource_keys={"duckdb"})
def print_continent_population(context, continent_population: SQL):
    context.log.info(f"Final asset:")
    context.log.info(context.resources.duckdb.query(continent_population))