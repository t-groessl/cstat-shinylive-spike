import io
import sys
from pathlib import Path

import duckdb
import pandas as pd
import plotly
import plotly.express as px
import pyarrow
import pyarrow.parquet as pq
import sklearn
from shiny import reactive
from shiny.express import input, render, ui
from shinywidgets import render_widget
from sklearn.cluster import KMeans

DATA = Path(__file__).parent / "data" / "sample.parquet"

ui.page_opts(title="Shinylive stack spike (pandas + duckdb)")
ui.input_slider("k", "Cluster (k)", 2, 6, 3)


@reactive.calc
def data() -> pd.DataFrame:
    return pq.read_table(DATA).to_pandas()


@render.text
def versions():
    df = data()
    agg = duckdb.sql(
        "select train_type, round(avg(delay),1) as mean_delay, count(*) as n "
        "from df group by 1 order by 1"
    ).df()
    n_duck = duckdb.sql(f"select count(*) from read_parquet('{DATA.as_posix()}')").fetchone()[0]
    return (
        f"python {sys.version.split()[0]} platform={sys.platform}\n"
        f"pandas {pd.__version__} | duckdb {duckdb.__version__} | pyarrow {pyarrow.__version__} | "
        f"sklearn {sklearn.__version__} | plotly {plotly.__version__}\n"
        f"rows={len(df)} | duckdb read_parquet rows={n_duck}\n{agg.to_string(index=False)}"
    )


@render.text
async def fetched():
    if sys.platform != "emscripten":
        return "not in browser"
    from pyodide.http import pyfetch
    from js import location
    href = str(location.href)
    base = href.split("/shinylive/")[0] if "/shinylive/" in href else str(location.origin)
    r = await pyfetch(base + "/data/sample.parquet")
    b = await r.bytes()
    df = pq.read_table(io.BytesIO(b)).to_pandas()
    return f"FETCH OK: {len(b)} bytes, {len(df)} rows via pyfetch + pyarrow"


@render_widget
def plot():
    df = data().copy()
    km = KMeans(n_clusters=input.k(), n_init=3, random_state=0).fit(df[["hour", "delay"]].to_numpy())
    df["cluster"] = km.labels_.astype(str)
    return px.scatter(df, x="hour", y="delay", color="cluster", title="KMeans + plotly in the browser")
