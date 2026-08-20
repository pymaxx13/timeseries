import json

file_path = r"c:\Users\Anusha\engression\engression-ts\engressionts\experiments\solar\baselines-solar-darts.ipynb"

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cell = {
   "cell_type": "code",
   "execution_count": None,
   "id": "added_train_ts",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ============================================================\n",
    "# CELL 7 — BUILD DARTS MULTIVARIATE TIMESERIES\n",
    "# ============================================================\n",
    "\n",
    "def gluonts_item_to_darts_mv(item, freq: str) -> TimeSeries:\n",
    "    start = item[\"start\"].to_timestamp() if hasattr(item[\"start\"], \"to_timestamp\") else pd.Timestamp(item[\"start\"])\n",
    "    target = np.asarray(item[\"target\"])\n",
    "    if target.ndim != 2:\n",
    "        raise ValueError(f\"Expected multivariate target with ndim=2, got shape {target.shape}\")\n",
    "    \n",
    "    values = target.T\n",
    "    times = pd.date_range(start=start, periods=values.shape[0], freq=freq)\n",
    "    cols = [f\"dim_{i}\" for i in range(values.shape[1])]\n",
    "    return TimeSeries.from_times_and_values(times, values, columns=cols)\n",
    "\n",
    "freq = ds.metadata.freq\n",
    "target_dim = int(ds.metadata.feat_static_cat[0].cardinality)\n",
    "train_grouper = MultivariateGrouper(max_target_dim=target_dim)\n",
    "train_mv_items = list(train_grouper(list(ds.train)))\n",
    "train_ts = gluonts_item_to_darts_mv(train_mv_items[0], freq)\n",
    "\n",
    "print(\"Train TS length:\", len(train_ts))\n"
   ]
}

insert_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and "ds = get_dataset(\n" in "".join(cell['source']):
        insert_idx = i + 1
        break

if insert_idx != -1:
    # check if it is already added
    already_added = False
    for cell in nb['cells']:
        if "gluonts_item_to_darts_mv" in "".join(cell.get('source', [])):
            already_added = True
            break
            
    if not already_added:
        nb['cells'].insert(insert_idx, new_cell)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print("Successfully inserted the missing cell into the notebook!")
    else:
        print("Cell already exists!")
else:
    print("Could not find the location to insert the cell.")
