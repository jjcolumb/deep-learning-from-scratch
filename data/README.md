# Data

`penguins.csv`: [Palmer Penguins](https://allisonhorst.github.io/palmerpenguins/),
Horst AM, Hill AP, Gorman KB (2020). Released under CC0. Collected by Dr.
Kristen Gorman at Palmer Station, Antarctica LTER.

`cuba.csv`, `cuba_outline.json`: 2,000 random points around Cuba labelled land
or sea, and the coastline used to label and plot them. Coastline from
[Natural Earth](https://www.naturalearthdata.com/) 1:50m countries (public
domain). Rebuild with `python make_cuba.py ne_50m_admin_0_countries.geojson`.

`frogs_train.bin`, `frogs_test.bin` (not in the repo, build them with
`python make_frogs.py`): every frog in [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
plus as many other photos. Krizhevsky A (2009), *Learning Multiple Layers of
Features from Tiny Images*.
