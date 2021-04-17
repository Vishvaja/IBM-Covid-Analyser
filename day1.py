import nest_asyncio
nest_asyncio.apply()

import twint

c = twint.Config()
c.Search = "coronavirus"

c.Since="2020-07-05 00:00:00"
c.Until="2020-07-06 00:00:00"
c.Limit=200
c.verified=True
c.Store_csv = True
c.Output = "tweet5"
twint.run.Search(c)
