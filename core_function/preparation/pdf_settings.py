import os


COUNTRY_LIST = ['Mexico', 'US, CA', 'EU', 'United States/ Canada', 'United States; Mexico; Canada; European Union, Japan, India', 'United States; Mexico; India; Canada; Japan; European Union; China', 'EU,US,CAN', 'United States; Mexico; Canada; Japan; European, India', 'US', 'United States;', 'France; Germany; Italy; Spain; United Kingdom', 'United States; Mexico; Canada; Japan; European Union; India; Turkey', 'United States; Mexico; Canada; Japan; European Union; India;', 'European Union', 'EU', 'EU,US,CA', 'United States; Mexico; Canada; Japan; European Union; India;  Turkey', 'United States; Mexico; Canada; Japan; European Union; India', 'United States; Mexico; Canada; Japan; European Union; India; Singapore, Australia', 'Canada', 'GE', 'United States; Mexico; Canada; Japan; European Union; India; ,', 'United States; Canada; European Union, Mexico, Japan, India', 'US, CA, JP, EU, IN', 'United States;', 'United States (California market only)', 'EU, IN, US', 'FR', 'United States', 'United States; Mexico; Canada;', 'United States; Mexico; Canada; Japan; European Union; India; China, Turkey', 'DE', 'United States; Mexico; Canada; Japan; European Union; India; , Turkey', 'US, CA, MX', 'Canada; European Union', 'united states', 'United Kingdom', 'United States; Mexico; Canada; Japan; European Union; India; China;Turkey', 'CA, CN, EU, IN, JP, MX, US', 'EU, CA, US, JP, MX, IN', 'United States; Canada', 'United States-California', 'US,CAN,MX', 'United States (California market only)', 'United States; Mexico; Canada; ; European Union; India;', 'Canada; European Union; Japan; Mexico; United States', 'United States', 'United States; Mexico; Canada; European Union', 'England', 'United States; Mexico; Canada; Japan; European Union;', 'United States; Mexico; Canada, European union, India, Japan', 'United States; Mexico; Canada; Japan; European Union; India; ; Turkey', 'United States; Mexico; Canada; Japan; European Union; India; china', 'Canada; European Union; India; Mexico; Japan; United States', 'Canada; European Union; Japan; United States', 'CA, US, MX, JP', 'United States', 'United States; Mexico; Canada; Japan; European Union; India;', 'Mexico, Japan, India, Canada, European Union', 'IT', 'United States; Mexico; Canada; Japan; European Union; India;', 'All', 'United States; Mexico; Canada; Japan; European Union; India; ;', 'United States; Mexico; Canada; Japan; European Union; China', 'Germany', 'European Union, Turkey', 'Australia', 'CA', 'India', 'UK', 'EU, JP, MX, US', 'all', 'United States; Mexico; Canada;  European Union;', 'United States; Mexico; Canada;', 'EU, IN, JP', 'EU, IN', 'EU/JP/MX', 'United States; Mexico; India; Canada; Japan; European Union;', 'United States, Canada, Mexico', 'CA', 'United States; Canada; Japan; European Union;', 'United States; Mexico; Canada; Japan; European Union; India; ;', 'Canada', 'US/CA', 'United States; Mexico; Canada; Japan; European Union; India,', 'EU, JP, MX', 'US/Canada', 'United States; Mexico; Canada; Japan; European Union; India; china; Turkey', 'United States; Mexico; Canada; Japan; European Union; India; china, Turkey', 'United States; Mexico; Canada; Japan; European Union; India', 'United States (California)', 'European Union,', 'United States; Canada; European Union; UK', 'United States; Mexico; Canada; Japan; European Union; India; China;', 'United States; Mexico; Canada; Japan; European Union; India; China', 'United States; Mexico; Canada; Japan; European Union; India; China', 'United States; Mexico; Canada; Japan; European Union;India; Turkey', 'Italy', 'Canada (British Columbia)', 'Canada; European Union; India; Japan; Mexico; United States', 'Canada; China; European Union; India; Japan; Mexico; United States', 'United States;', 'Canada; Mexico; United States', 'MX']





