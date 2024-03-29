# Initialize some mappings

month_mapping = {"":"",
            "January":"January",
            "February":"February",
            "March":"March",
            "April":"April",
            "May":"May",
            "June":"June",
            "July":"July",
            "August":"August",
            "September":"September",
            "October":"October",
            "November":"November",
            "December":"December"}

iucn_status_mapping = {
            "": "",
            "Extinct (EX)": "EX",
            "Extinct in the Wild (EW)": "EW",
            "Critically Endangered (CR)": "CR",
            "Endangered (EN)": "EN",
            "Vulnerable (VU)": "VU",
            "Near Threatened (NT)": "NT",
            "Least Concern (LC)": "LC",
            "Data Deficient (DD)": "DD",
            "Not Evaluated (NE)": "NE"
        }

type_status_mapping = {
            "": "",
            "Allolectotype": "ALLOLECTOTYPE",
            "Alloneotype": "ALLONEOTYPE",
            "Allotype": "ALLOTYPE",
            "Cotype": "COTYPE",
            "Epitype": "EPITYPE",
            "Exepitype": "EXEPITYPE",
            "Exholotype": "EXHOLOTYPE",
            "Exisotype": "EXISOTYPE",
            "Exlectotype": "EXLECTOTYPE",
            "Exneotype": "EXNEOTYPE",
            "Exparatype": "EXPARATYPE",
            "Exsyntype": "EXSYNTYPE",
            "Extype": "EXTYPE",
            "Hapantotype": "HAPANTOTYPE",
            "Holotype": "HOLOTYPE",
            "Hypotype": "HYPOTYPE",
            "Iconotype": "ICONOTYPE",
            "Isolectotype": "ISOLECTOTYPE",
            "Isoneotype": "ISONEOTYPE",
            "Isoparatype": "ISOPARATYPE",
            "Isosyntype": "ISOSYNTYPE",
            "Isotype": "ISOTYPE",
            "Lectotype": "LECTOTYPE",
            "Neotype": "NEOTYPE",
            "Notatype": "NOTATYPE",
            "Original material": "ORIGINALMATERIAL",
            "Paralectotype": "PARALECTOTYPE",
            "Paraneotype": "PARANEOTYPE",
            "Paratype": "PARATYPE",
            "Plastoholotype": "PLASTOHOLOTYPE",
            "Plastoisotype": "PLASTOISOTYPE",
            "Plastolectotype": "PLASTOLECTOTYPE",
            "Plastoneotype": "PLASTONEOTYPE",
            "Plastoparatype": "PLASTOPARATYPE",
            "Plastosyntype": "PLASTOSYNTYPE",
            "Plastotype": "PLASTOTYPE",
            "Plesiotype": "PLESIOTYPE",
            "Secondary type": "SECONDARYTYPE",
            "Supplementary type": "SUPPLEMENTARYTYPE",
            "Syntype": "SYNTYPE",
            "Topotype": "TOPOTYPE",
            "Type": "TYPE",
            "Type genus": "TYPE_GENUS",
            "Type species": "TYPE_SPECIES",
        }

continent_mapping = {
            "": "",
            "Africa": "AFRICA",
            "Antarctica": "ANTARCTICA",
            "Asia": "ASIA",
            "Europe": "EUROPE",
            "North America": "NORTH_AMERICA",
            "Oceania": "OCEANIA",
            "South America": "SOUTH_AMERICA",
        }

country_mapping = {
            "": "",
            "Afghanistan": "AFGHANISTAN",
            "Åland Islands": "ALAND_ISLANDS",
            "Albania": "ALBANIA",
            "Algeria": "ALGERIA",
            "American Samoa": "AMERICAN_SAMOA",
            "Andorra": "ANDORRA",
            "Angola": "ANGOLA",
            "Anguilla": "ANGUILLA",
            "Antarctica": "ANTARCTICA",
            "Antigua and Barbuda": "ANTIGUA_BARBUDA",
            "Argentina": "ARGENTINA",
            "Armenia": "ARMENIA",
            "Aruba": "ARUBA",
            "Australia": "AUSTRALIA",
            "Austria": "AUSTRIA",
            "Azerbaijan": "AZERBAIJAN",
            "Bahamas": "BAHAMAS",
            "Bahrain": "BAHRAIN",
            "Bangladesh": "BANGLADESH",
            "Barbados": "BARBADOS",
            "Belarus": "BELARUS",
            "Belgium": "BELGIUM",
            "Belize": "BELIZE",
            "Benin": "BENIN",
            "Bermuda": "BERMUDA",
            "Bhutan": "BHUTAN",
            "Bolivia (Plurinational State of)": "BOLIVIA",
            "Bonaire, Sint Eustatius and Saba": "BONAIRE_SINT_EUSTATIUS_SABA",
            "Bosnia and Herzegovina": "BOSNIA_HERZEGOVINA",
            "Botswana": "BOTSWANA",
            "Bouvet Island": "BOUVET_ISLAND",
            "Brazil": "BRAZIL",
            "British Indian Ocean Territory": "BRITISH_INDIAN_OCEAN_TERRITORY",
            "Brunei Darussalam": "BRUNEI_DARUSSALAM",
            "Bulgaria": "BULGARIA",
            "Burkina Faso": "BURKINA_FASO",
            "Burundi": "BURUNDI",
            "Cambodia": "CAMBODIA",
            "Cameroon": "CAMEROON",
            "Canada": "CANADA",
            "Cabo Verde": "CAPE_VERDE",
            "Cayman Islands": "CAYMAN_ISLANDS",
            "Central African Republic": "CENTRAL_AFRICAN_REPUBLIC",
            "Chad": "CHAD",
            "Chile": "CHILE",
            "China": "CHINA",
            "Christmas Island": "CHRISTMAS_ISLAND",
            "Cocos (Keeling) Islands": "COCOS_ISLANDS",
            "Colombia": "COLOMBIA",
            "Comoros": "COMOROS",
            "Congo, Republic of the": "CONGO",
            "Congo, Democratic Republic of the": "CONGO_DEMOCRATIC_REPUBLIC",
            "Cook Islands": "COOK_ISLANDS",
            "Costa Rica": "COSTA_RICA",
            "Côte d’Ivoire": "CÔTE_DIVOIRE",
            "Croatia": "CROATIA",
            "Cuba": "CUBA",
            "Curaçao": "CURAÇAO",
            "Cyprus": "CYPRUS",
            "Czechia": "CZECH_REPUBLIC",
            "Denmark": "DENMARK",
            "Djibouti": "DJIBOUTI",
            "Dominica": "DOMINICA",
            "Dominican Republic": "DOMINICAN_REPUBLIC",
            "Ecuador": "ECUADOR",
            "Egypt": "EGYPT",
            "El Salvador": "EL_SALVADOR",
            "Equatorial Guinea": "EQUATORIAL_GUINEA",
            "Eritrea": "ERITREA",
            "Estonia": "ESTONIA",
            "Ethiopia": "ETHIOPIA",
            "Falkland Islands (Malvinas)": "FALKLAND_ISLANDS",
            "Faroe Islands": "FAROE_ISLANDS",
            "Fiji": "FIJI",
            "Finland": "FINLAND",
            "France": "FRANCE",
            "French Guiana": "FRENCH_GUIANA",
            "French Polynesia": "FRENCH_POLYNESIA",
            "French Southern Territories": "FRENCH_SOUTHERN_TERRITORIES",
            "Gabon": "GABON",
            "Gambia": "GAMBIA",
            "Georgia": "GEORGIA",
            "Germany": "GERMANY",
            "Ghana": "GHANA",
            "Gibraltar": "GIBRALTAR",
            "Greece": "GREECE",
            "Greenland": "GREENLAND",
            "Grenada": "GRENADA",
            "Guadeloupe": "GUADELOUPE",
            "Guam": "GUAM",
            "Guatemala": "GUATEMALA",
            "Guernsey": "GUERNSEY",
            "Guinea": "GUINEA",
            "Guinea-Bissau": "GUINEA_BISSAU",
            "Guyana": "GUYANA",
            "Haiti": "HAITI",
            "Heard Island and McDonald Islands": "HEARD_MCDONALD_ISLANDS",
            "Honduras": "HONDURAS",
            "Hong Kong": "HONG_KONG",
            "Hungary": "HUNGARY",
            "Iceland": "ICELAND",
            "India": "INDIA",
            "Indonesia": "INDONESIA",
            "International Waters": "INTERNATIONAL_WATERS",
            "Iran (Islamic Republic of)": "IRAN",
            "Iraq": "IRAQ",
            "Ireland": "IRELAND",
            "Isle of Man": "ISLE_OF_MAN",
            "Israel": "ISRAEL",
            "Italy": "ITALY",
            "Jamaica": "JAMAICA",
            "Japan": "JAPAN",
            "Jersey": "JERSEY",
            "Jordan": "JORDAN",
            "Kazakhstan": "KAZAKHSTAN",
            "Kenya": "KENYA",
            "Kiribati": "KIRIBATI",
            "Korea, Republic of": "KOREA_SOUTH",
            "Kosovo": "KOSOVO",
            "Kuwait": "KUWAIT",
            "Kyrgyzstan": "KYRGYZSTAN",
            "Lao Peoples Democratic Republic": "LAO",
            "Latvia": "LATVIA",
            "Lebanon": "LEBANON",
            "Lesotho": "LESOTHO",
            "Liberia": "LIBERIA",
            "Libya": "LIBYA",
            "Liechtenstein": "LIECHTENSTEIN",
            "Lithuania": "LITHUANIA",
            "Luxembourg": "LUXEMBOURG",
            "Macao": "MACAO",
            "North Macedonia": "MACEDONIA",
            "Madagascar": "MADAGASCAR",
            "Malawi": "MALAWI",
            "Malaysia": "MALAYSIA",
            "Maldives": "MALDIVES",
            "Mali": "MALI",
            "Malta": "MALTA",
            "Marshall Islands": "MARSHALL_ISLANDS",
            "Martinique": "MARTINIQUE",
            "Mauritania": "MAURITANIA",
            "Mauritius": "MAURITIUS",
            "Mayotte": "MAYOTTE",
            "Mexico": "MEXICO",
            "Micronesia (Federated States of)": "MICRONESIA",
            "Moldova, Republic of": "MOLDOVA",
            "Monaco": "MONACO",
            "Mongolia": "MONGOLIA",
            "Montenegro": "MONTENEGRO",
            "Montserrat": "MONTSERRAT",
            "Morocco": "MOROCCO",
            "Mozambique": "MOZAMBIQUE",
            "Myanmar": "MYANMAR",
            "Namibia": "NAMIBIA",
            "Nauru": "NAURU",
            "Nepal": "NEPAL",
            "Netherlands": "NETHERLANDS",
            "New Caledonia": "NEW_CALEDONIA",
            "New Zealand": "NEW_ZEALAND",
            "Nicaragua": "NICARAGUA",
            "Niger": "NIGER",
            "Nigeria": "NIGERIA",
            "Niue": "NIUE",
            "Norfolk Island": "NORFOLK_ISLAND",
            "Northern Mariana Islands": "NORTHERN_MARIANA_ISLANDS",
            "Norway": "NORWAY",
            "Oman": "OMAN",
            "Pakistan": "PAKISTAN",
            "Palau": "PALAU",
            "Palestine, State of": "PALESTINIAN_TERRITORY",
            "Panama": "PANAMA",
            "Papua New Guinea": "PAPUA_NEW_GUINEA",
            "Paraguay": "PARAGUAY",
            "Peru": "PERU",
            "Philippines": "PHILIPPINES",
            "Pitcairn": "PITCAIRN",
            "Poland": "POLAND",
            "Portugal": "PORTUGAL",
            "Puerto Rico": "PUERTO_RICO",
            "Qatar": "QATAR",
            "Réunion": "RÉUNION",
            "Romania": "ROMANIA",
            "Russian Federation": "RUSSIAN_FEDERATION",
            "Rwanda": "RWANDA",
            "Saint Barthélemy": "SAINT_BARTHÉLEMY",
            "Saint Helena, Ascension and Tristan da Cunha": "SAINT_HELENA_ASCENSION_TRISTAN_DA_CUNHA",
            "Saint Kitts and Nevis": "SAINT_KITTS_NEVIS",
            "Saint Lucia": "SAINT_LUCIA",
            "Saint Martin (French part)": "SAINT_MARTIN_FRENCH",
            "Saint Pierre and Miquelon": "SAINT_PIERRE_MIQUELON",
            "Saint Vincent and the Grenadines": "SAINT_VINCENT_GRENADINES",
            "Samoa": "SAMOA",
            "San Marino": "SAN_MARINO",
            "Sao Tome and Principe": "SAO_TOME_PRINCIPE",
            "Saudi Arabia": "SAUDI_ARABIA",
            "Senegal": "SENEGAL",
            "Serbia": "SERBIA",
            "Seychelles": "SEYCHELLES",
            "Sierra Leone": "SIERRA_LEONE",
            "Singapore": "SINGAPORE",
            "Sint Maarten (Dutch part)": "SINT_MAARTEN",
            "Slovakia": "SLOVAKIA",
            "Slovenia": "SLOVENIA",
            "Solomon Islands": "SOLOMON_ISLANDS",
            "Somalia": "SOMALIA",
            "South Africa": "SOUTH_AFRICA",
            "South Georgia and the South Sandwich Islands": "SOUTH_GEORGIA_SANDWICH_ISLANDS",
            "South Sudan": "SOUTH_SUDAN",
            "Spain": "SPAIN",
            "Sri Lanka": "SRI_LANKA",
            "Sudan": "SUDAN",
            "Suriname": "SURINAME",
            "Svalbard and Jan Mayen": "SVALBARD_JAN_MAYEN",
            "Swaziland": "SWAZILAND",
            "Sweden": "SWEDEN",
            "Switzerland": "SWITZERLAND",
            "Syrian Arab Republic": "SYRIA",
            "Taiwan, Province of China": "TAIWAN",
            "Tajikistan": "TAJIKISTAN",
            "Tanzania, United Republic of": "TANZANIA",
            "Thailand": "THAILAND",
            "Timor-Leste": "TIMOR_LESTE",
            "Togo": "TOGO",
            "Tokelau": "TOKELAU",
            "Tonga": "TONGA",
            "Trinidad and Tobago": "TRINIDAD_TOBAGO",
            "Tunisia": "TUNISIA",
            "Türkiye": "TURKEY",
            "Turkmenistan": "TURKMENISTAN",
            "Turks and Caicos Islands": "TURKS_CAICOS_ISLANDS",
            "Tuvalu": "TUVALU",
            "Uganda": "UGANDA",
            "Ukraine": "UKRAINE",
            "United Arab Emirates": "UNITED_ARAB_EMIRATES",
            "United Kingdom of Great Britain and Northern Ireland": "UNITED_KINGDOM",
            "United States of America": "UNITED_STATES",
            "United States Minor Outlying Islands": "UNITED_STATES_OUTLYING_ISLANDS",
            "Unknown or Invalid territory": "UNKNOWN",
            "Uruguay": "URUGUAY",
            "Uzbekistan": "UZBEKISTAN",
            "Vanuatu": "VANUATU",
            "Holy See": "VATICAN",
            "Venezuela (Bolivarian Republic of)": "VENEZUELA",
            "Viet Nam": "VIETNAM",
            "Virgin Islands, (U.S.)": "VIRGIN_ISLANDS",
            "Virgin Islands (British)": "VIRGIN_ISLANDS_BRITISH",
            "Wallis and Futuna": "WALLIS_FUTUNA",
            "Western Sahara": "WESTERN_SAHARA",
            "Yemen": "YEMEN",
            "Zambia": "ZAMBIA",
            "Zimbabwe": "ZIMBABWE"
        }