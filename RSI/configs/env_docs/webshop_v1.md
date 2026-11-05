# WebShop runtime interface (v1, probed from the live environment)

You shop on a text e-commerce site to satisfy the instruction (e.g.
"i need a long lasting, anti-dandruff shampoo, and price lower than 50.00
dollars"). Each turn is exactly one action.

## Actions (the only two tools)

- `search` — free-text keyword query (e.g. "shampoo anti dandruff"). Starts a
  new search; the response is a results page listing products with titles and
  prices.
- `click` — click a button/link/option by its EXACT visible text. Typical
  clicks:
  - a product title (with its `[id]`) to open the item page;
  - option buttons on the item page (size / color / capacity — clicking an
    option filters the product; click the title again to refresh with the
    option applied);
  - `buy now` — purchases the currently configured product and ENDS the
    episode. Only click it when the visible product + options match every
    instruction constraint (attributes and price cap);
  - `back with search` — return to the results page.

## Reading the page

Search results show `[pid] title $price` per item plus a pagination footer
(`page 1 of N (total ... items); next page ...`). The item page shows the
full description, selected options, and the buy buttons. The instruction's
constraints (attributes, size, price ceiling) are the scoring target —
the closer the purchase matches, the higher the reward.

## Strategy

Refine keywords and filters when results don't match; open candidates to
inspect options; never `buy now` while a constraint is visibly unmet. There
is no undo after purchase, so verify on the item page first.
