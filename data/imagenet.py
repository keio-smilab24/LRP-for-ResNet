import os
from typing import Callable, Literal, Optional

import torchvision
from torch.utils.data import Dataset


class ImageNetDataset(Dataset):
    """This dataset does not support training set, only test set."""
    def __init__(
        self, root: str, image_set: Literal["train", "test"] = "test", transform: Optional[Callable] = None
    ) -> None:
        super().__init__()

        self.root = root
        self.image_set = image_set
        self.transform = transform

        if image_set == "test":
            data_path = os.path.join(root, "imagenet")
            self.instance = torchvision.datasets.ImageNet(data_path, split="val", transform=transform)
        elif image_set == "train":
            self.instance = None
        else:
            raise ValueError(f"image_set must be 'train' or 'test', but got {image_set}")

    def __getitem__(self, index):
        if self.image_set == "test":
            return self.instance[index]
        else:
            return -1, None

    def __len__(self) -> int:
        if self.image_set == "test":
            return len(self.instance)
        else:
            return 0


# Following list is based on https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json
IMAGENET_CLASSES = [
    "tench:n01440764",
    "goldfish:n01443537",
    "great_white_shark:n01484850",
    "tiger_shark:n01491361",
    "hammerhead:n01494475",
    "electric_ray:n01496331",
    "stingray:n01498041",
    "cock:n01514668",
    "hen:n01514859",
    "ostrich:n01518878",
    "brambling:n01530575",
    "goldfinch:n01531178",
    "house_finch:n01532829",
    "junco:n01534433",
    "indigo_bunting:n01537544",
    "robin:n01558993",
    "bulbul:n01560419",
    "jay:n01580077",
    "magpie:n01582220",
    "chickadee:n01592084",
    "water_ouzel:n01601694",
    "kite:n01608432",
    "bald_eagle:n01614925",
    "vulture:n01616318",
    "great_grey_owl:n01622779",
    "European_fire_salamander:n01629819",
    "common_newt:n01630670",
    "eft:n01631663",
    "spotted_salamander:n01632458",
    "axolotl:n01632777",
    "bullfrog:n01641577",
    "tree_frog:n01644373",
    "tailed_frog:n01644900",
    "loggerhead:n01664065",
    "leatherback_turtle:n01665541",
    "mud_turtle:n01667114",
    "terrapin:n01667778",
    "box_turtle:n01669191",
    "banded_gecko:n01675722",
    "common_iguana:n01677366",
    "American_chameleon:n01682714",
    "whiptail:n01685808",
    "agama:n01687978",
    "frilled_lizard:n01688243",
    "alligator_lizard:n01689811",
    "Gila_monster:n01692333",
    "green_lizard:n01693334",
    "African_chameleon:n01694178",
    "Komodo_dragon:n01695060",
    "African_crocodile:n01697457",
    "American_alligator:n01698640",
    "triceratops:n01704323",
    "thunder_snake:n01728572",
    "ringneck_snake:n01728920",
    "hognose_snake:n01729322",
    "green_snake:n01729977",
    "king_snake:n01734418",
    "garter_snake:n01735189",
    "water_snake:n01737021",
    "vine_snake:n01739381",
    "night_snake:n01740131",
    "boa_constrictor:n01742172",
    "rock_python:n01744401",
    "Indian_cobra:n01748264",
    "green_mamba:n01749939",
    "sea_snake:n01751748",
    "horned_viper:n01753488",
    "diamondback:n01755581",
    "sidewinder:n01756291",
    "trilobite:n01768244",
    "harvestman:n01770081",
    "scorpion:n01770393",
    "black_and_gold_garden_spider:n01773157",
    "barn_spider:n01773549",
    "garden_spider:n01773797",
    "black_widow:n01774384",
    "tarantula:n01774750",
    "wolf_spider:n01775062",
    "tick:n01776313",
    "centipede:n01784675",
    "black_grouse:n01795545",
    "ptarmigan:n01796340",
    "ruffed_grouse:n01797886",
    "prairie_chicken:n01798484",
    "peacock:n01806143",
    "quail:n01806567",
    "partridge:n01807496",
    "African_grey:n01817953",
    "macaw:n01818515",
    "sulphur-crested_cockatoo:n01819313",
    "lorikeet:n01820546",
    "coucal:n01824575",
    "bee_eater:n01828970",
    "hornbill:n01829413",
    "hummingbird:n01833805",
    "jacamar:n01843065",
    "toucan:n01843383",
    "drake:n01847000",
    "red-breasted_merganser:n01855032",
    "goose:n01855672",
    "black_swan:n01860187",
    "tusker:n01871265",
    "echidna:n01872401",
    "platypus:n01873310",
    "wallaby:n01877812",
    "koala:n01882714",
    "wombat:n01883070",
    "jellyfish:n01910747",
    "sea_anemone:n01914609",
    "brain_coral:n01917289",
    "flatworm:n01924916",
    "nematode:n01930112",
    "conch:n01943899",
    "snail:n01944390",
    "slug:n01945685",
    "sea_slug:n01950731",
    "chiton:n01955084",
    "chambered_nautilus:n01968897",
    "Dungeness_crab:n01978287",
    "rock_crab:n01978455",
    "fiddler_crab:n01980166",
    "king_crab:n01981276",
    "American_lobster:n01983481",
    "spiny_lobster:n01984695",
    "crayfish:n01985128",
    "hermit_crab:n01986214",
    "isopod:n01990800",
    "white_stork:n02002556",
    "black_stork:n02002724",
    "spoonbill:n02006656",
    "flamingo:n02007558",
    "little_blue_heron:n02009229",
    "American_egret:n02009912",
    "bittern:n02011460",
    "crane:n02012849",
    "limpkin:n02013706",
    "European_gallinule:n02017213",
    "American_coot:n02018207",
    "bustard:n02018795",
    "ruddy_turnstone:n02025239",
    "red-backed_sandpiper:n02027492",
    "redshank:n02028035",
    "dowitcher:n02033041",
    "oystercatcher:n02037110",
    "pelican:n02051845",
    "king_penguin:n02056570",
    "albatross:n02058221",
    "grey_whale:n02066245",
    "killer_whale:n02071294",
    "dugong:n02074367",
    "sea_lion:n02077923",
    "Chihuahua:n02085620",
    "Japanese_spaniel:n02085782",
    "Maltese_dog:n02085936",
    "Pekinese:n02086079",
    "Shih-Tzu:n02086240",
    "Blenheim_spaniel:n02086646",
    "papillon:n02086910",
    "toy_terrier:n02087046",
    "Rhodesian_ridgeback:n02087394",
    "Afghan_hound:n02088094",
    "basset:n02088238",
    "beagle:n02088364",
    "bloodhound:n02088466",
    "bluetick:n02088632",
    "black-and-tan_coonhound:n02089078",
    "Walker_hound:n02089867",
    "English_foxhound:n02089973",
    "redbone:n02090379",
    "borzoi:n02090622",
    "Irish_wolfhound:n02090721",
    "Italian_greyhound:n02091032",
    "whippet:n02091134",
    "Ibizan_hound:n02091244",
    "Norwegian_elkhound:n02091467",
    "otterhound:n02091635",
    "Saluki:n02091831",
    "Scottish_deerhound:n02092002",
    "Weimaraner:n02092339",
    "Staffordshire_bullterrier:n02093256",
    "American_Staffordshire_terrier:n02093428",
    "Bedlington_terrier:n02093647",
    "Border_terrier:n02093754",
    "Kerry_blue_terrier:n02093859",
    "Irish_terrier:n02093991",
    "Norfolk_terrier:n02094114",
    "Norwich_terrier:n02094258",
    "Yorkshire_terrier:n02094433",
    "wire-haired_fox_terrier:n02095314",
    "Lakeland_terrier:n02095570",
    "Sealyham_terrier:n02095889",
    "Airedale:n02096051",
    "cairn:n02096177",
    "Australian_terrier:n02096294",
    "Dandie_Dinmont:n02096437",
    "Boston_bull:n02096585",
    "miniature_schnauzer:n02097047",
    "giant_schnauzer:n02097130",
    "standard_schnauzer:n02097209",
    "Scotch_terrier:n02097298",
    "Tibetan_terrier:n02097474",
    "silky_terrier:n02097658",
    "soft-coated_wheaten_terrier:n02098105",
    "West_Highland_white_terrier:n02098286",
    "Lhasa:n02098413",
    "flat-coated_retriever:n02099267",
    "curly-coated_retriever:n02099429",
    "golden_retriever:n02099601",
    "Labrador_retriever:n02099712",
    "Chesapeake_Bay_retriever:n02099849",
    "German_short-haired_pointer:n02100236",
    "vizsla:n02100583",
    "English_setter:n02100735",
    "Irish_setter:n02100877",
    "Gordon_setter:n02101006",
    "Brittany_spaniel:n02101388",
    "clumber:n02101556",
    "English_springer:n02102040",
    "Welsh_springer_spaniel:n02102177",
    "cocker_spaniel:n02102318",
    "Sussex_spaniel:n02102480",
    "Irish_water_spaniel:n02102973",
    "kuvasz:n02104029",
    "schipperke:n02104365",
    "groenendael:n02105056",
    "malinois:n02105162",
    "briard:n02105251",
    "kelpie:n02105412",
    "komondor:n02105505",
    "Old_English_sheepdog:n02105641",
    "Shetland_sheepdog:n02105855",
    "collie:n02106030",
    "Border_collie:n02106166",
    "Bouvier_des_Flandres:n02106382",
    "Rottweiler:n02106550",
    "German_shepherd:n02106662",
    "Doberman:n02107142",
    "miniature_pinscher:n02107312",
    "Greater_Swiss_Mountain_dog:n02107574",
    "Bernese_mountain_dog:n02107683",
    "Appenzeller:n02107908",
    "EntleBucher:n02108000",
    "boxer:n02108089",
    "bull_mastiff:n02108422",
    "Tibetan_mastiff:n02108551",
    "French_bulldog:n02108915",
    "Great_Dane:n02109047",
    "Saint_Bernard:n02109525",
    "Eskimo_dog:n02109961",
    "malamute:n02110063",
    "Siberian_husky:n02110185",
    "dalmatian:n02110341",
    "affenpinscher:n02110627",
    "basenji:n02110806",
    "pug:n02110958",
    "Leonberg:n02111129",
    "Newfoundland:n02111277",
    "Great_Pyrenees:n02111500",
    "Samoyed:n02111889",
    "Pomeranian:n02112018",
    "chow:n02112137",
    "keeshond:n02112350",
    "Brabancon_griffon:n02112706",
    "Pembroke:n02113023",
    "Cardigan:n02113186",
    "toy_poodle:n02113624",
    "miniature_poodle:n02113712",
    "standard_poodle:n02113799",
    "Mexican_hairless:n02113978",
    "timber_wolf:n02114367",
    "white_wolf:n02114548",
    "red_wolf:n02114712",
    "coyote:n02114855",
    "dingo:n02115641",
    "dhole:n02115913",
    "African_hunting_dog:n02116738",
    "hyena:n02117135",
    "red_fox:n02119022",
    "kit_fox:n02119789",
    "Arctic_fox:n02120079",
    "grey_fox:n02120505",
    "tabby:n02123045",
    "tiger_cat:n02123159",
    "Persian_cat:n02123394",
    "Siamese_cat:n02123597",
    "Egyptian_cat:n02124075",
    "cougar:n02125311",
    "lynx:n02127052",
    "leopard:n02128385",
    "snow_leopard:n02128757",
    "jaguar:n02128925",
    "lion:n02129165",
    "tiger:n02129604",
    "cheetah:n02130308",
    "brown_bear:n02132136",
    "American_black_bear:n02133161",
    "ice_bear:n02134084",
    "sloth_bear:n02134418",
    "mongoose:n02137549",
    "meerkat:n02138441",
    "tiger_beetle:n02165105",
    "ladybug:n02165456",
    "ground_beetle:n02167151",
    "long-horned_beetle:n02168699",
    "leaf_beetle:n02169497",
    "dung_beetle:n02172182",
    "rhinoceros_beetle:n02174001",
    "weevil:n02177972",
    "fly:n02190166",
    "bee:n02206856",
    "ant:n02219486",
    "grasshopper:n02226429",
    "cricket:n02229544",
    "walking_stick:n02231487",
    "cockroach:n02233338",
    "mantis:n02236044",
    "cicada:n02256656",
    "leafhopper:n02259212",
    "lacewing:n02264363",
    "dragonfly:n02268443",
    "damselfly:n02268853",
    "admiral:n02276258",
    "ringlet:n02277742",
    "monarch:n02279972",
    "cabbage_butterfly:n02280649",
    "sulphur_butterfly:n02281406",
    "lycaenid:n02281787",
    "starfish:n02317335",
    "sea_urchin:n02319095",
    "sea_cucumber:n02321529",
    "wood_rabbit:n02325366",
    "hare:n02326432",
    "Angora:n02328150",
    "hamster:n02342885",
    "porcupine:n02346627",
    "fox_squirrel:n02356798",
    "marmot:n02361337",
    "beaver:n02363005",
    "guinea_pig:n02364673",
    "sorrel:n02389026",
    "zebra:n02391049",
    "hog:n02395406",
    "wild_boar:n02396427",
    "warthog:n02397096",
    "hippopotamus:n02398521",
    "ox:n02403003",
    "water_buffalo:n02408429",
    "bison:n02410509",
    "ram:n02412080",
    "bighorn:n02415577",
    "ibex:n02417914",
    "hartebeest:n02422106",
    "impala:n02422699",
    "gazelle:n02423022",
    "Arabian_camel:n02437312",
    "llama:n02437616",
    "weasel:n02441942",
    "mink:n02442845",
    "polecat:n02443114",
    "black-footed_ferret:n02443484",
    "otter:n02444819",
    "skunk:n02445715",
    "badger:n02447366",
    "armadillo:n02454379",
    "three-toed_sloth:n02457408",
    "orangutan:n02480495",
    "gorilla:n02480855",
    "chimpanzee:n02481823",
    "gibbon:n02483362",
    "siamang:n02483708",
    "guenon:n02484975",
    "patas:n02486261",
    "baboon:n02486410",
    "macaque:n02487347",
    "langur:n02488291",
    "colobus:n02488702",
    "proboscis_monkey:n02489166",
    "marmoset:n02490219",
    "capuchin:n02492035",
    "howler_monkey:n02492660",
    "titi:n02493509",
    "spider_monkey:n02493793",
    "squirrel_monkey:n02494079",
    "Madagascar_cat:n02497673",
    "indri:n02500267",
    "Indian_elephant:n02504013",
    "African_elephant:n02504458",
    "lesser_panda:n02509815",
    "giant_panda:n02510455",
    "barracouta:n02514041",
    "eel:n02526121",
    "coho:n02536864",
    "rock_beauty:n02606052",
    "anemone_fish:n02607072",
    "sturgeon:n02640242",
    "gar:n02641379",
    "lionfish:n02643566",
    "puffer:n02655020",
    "abacus:n02666196",
    "abaya:n02667093",
    "academic_gown:n02669723",
    "accordion:n02672831",
    "acoustic_guitar:n02676566",
    "aircraft_carrier:n02687172",
    "airliner:n02690373",
    "airship:n02692877",
    "altar:n02699494",
    "ambulance:n02701002",
    "amphibian:n02704792",
    "analog_clock:n02708093",
    "apiary:n02727426",
    "apron:n02730930",
    "ashcan:n02747177",
    "assault_rifle:n02749479",
    "backpack:n02769748",
    "bakery:n02776631",
    "balance_beam:n02777292",
    "balloon:n02782093",
    "ballpoint:n02783161",
    "Band_Aid:n02786058",
    "banjo:n02787622",
    "bannister:n02788148",
    "barbell:n02790996",
    "barber_chair:n02791124",
    "barbershop:n02791270",
    "barn:n02793495",
    "barometer:n02794156",
    "barrel:n02795169",
    "barrow:n02797295",
    "baseball:n02799071",
    "basketball:n02802426",
    "bassinet:n02804414",
    "bassoon:n02804610",
    "bathing_cap:n02807133",
    "bath_towel:n02808304",
    "bathtub:n02808440",
    "beach_wagon:n02814533",
    "beacon:n02814860",
    "beaker:n02815834",
    "bearskin:n02817516",
    "beer_bottle:n02823428",
    "beer_glass:n02823750",
    "bell_cote:n02825657",
    "bib:n02834397",
    "bicycle-built-for-two:n02835271",
    "bikini:n02837789",
    "binder:n02840245",
    "binoculars:n02841315",
    "birdhouse:n02843684",
    "boathouse:n02859443",
    "bobsled:n02860847",
    "bolo_tie:n02865351",
    "bonnet:n02869837",
    "bookcase:n02870880",
    "bookshop:n02871525",
    "bottlecap:n02877765",
    "bow:n02879718",
    "bow_tie:n02883205",
    "brass:n02892201",
    "brassiere:n02892767",
    "breakwater:n02894605",
    "breastplate:n02895154",
    "broom:n02906734",
    "bucket:n02909870",
    "buckle:n02910353",
    "bulletproof_vest:n02916936",
    "bullet_train:n02917067",
    "butcher_shop:n02927161",
    "cab:n02930766",
    "caldron:n02939185",
    "candle:n02948072",
    "cannon:n02950826",
    "canoe:n02951358",
    "can_opener:n02951585",
    "cardigan:n02963159",
    "car_mirror:n02965783",
    "carousel:n02966193",
    "carpenter's_kit:n02966687",
    "carton:n02971356",
    "car_wheel:n02974003",
    "cash_machine:n02977058",
    "cassette:n02978881",
    "cassette_player:n02979186",
    "castle:n02980441",
    "catamaran:n02981792",
    "CD_player:n02988304",
    "cello:n02992211",
    "cellular_telephone:n02992529",
    "chain:n02999410",
    "chainlink_fence:n03000134",
    "chain_mail:n03000247",
    "chain_saw:n03000684",
    "chest:n03014705",
    "chiffonier:n03016953",
    "chime:n03017168",
    "china_cabinet:n03018349",
    "Christmas_stocking:n03026506",
    "church:n03028079",
    "cinema:n03032252",
    "cleaver:n03041632",
    "cliff_dwelling:n03042490",
    "cloak:n03045698",
    "clog:n03047690",
    "cocktail_shaker:n03062245",
    "coffee_mug:n03063599",
    "coffeepot:n03063689",
    "coil:n03065424",
    "combination_lock:n03075370",
    "computer_keyboard:n03085013",
    "confectionery:n03089624",
    "container_ship:n03095699",
    "convertible:n03100240",
    "corkscrew:n03109150",
    "cornet:n03110669",
    "cowboy_boot:n03124043",
    "cowboy_hat:n03124170",
    "cradle:n03125729",
    "crane:n03126707",
    "crash_helmet:n03127747",
    "crate:n03127925",
    "crib:n03131574",
    "Crock_Pot:n03133878",
    "croquet_ball:n03134739",
    "crutch:n03141823",
    "cuirass:n03146219",
    "dam:n03160309",
    "desk:n03179701",
    "desktop_computer:n03180011",
    "dial_telephone:n03187595",
    "diaper:n03188531",
    "digital_clock:n03196217",
    "digital_watch:n03197337",
    "dining_table:n03201208",
    "dishrag:n03207743",
    "dishwasher:n03207941",
    "disk_brake:n03208938",
    "dock:n03216828",
    "dogsled:n03218198",
    "dome:n03220513",
    "doormat:n03223299",
    "drilling_platform:n03240683",
    "drum:n03249569",
    "drumstick:n03250847",
    "dumbbell:n03255030",
    "Dutch_oven:n03259280",
    "electric_fan:n03271574",
    "electric_guitar:n03272010",
    "electric_locomotive:n03272562",
    "entertainment_center:n03290653",
    "envelope:n03291819",
    "espresso_maker:n03297495",
    "face_powder:n03314780",
    "feather_boa:n03325584",
    "file:n03337140",
    "fireboat:n03344393",
    "fire_engine:n03345487",
    "fire_screen:n03347037",
    "flagpole:n03355925",
    "flute:n03372029",
    "folding_chair:n03376595",
    "football_helmet:n03379051",
    "forklift:n03384352",
    "fountain:n03388043",
    "fountain_pen:n03388183",
    "four-poster:n03388549",
    "freight_car:n03393912",
    "French_horn:n03394916",
    "frying_pan:n03400231",
    "fur_coat:n03404251",
    "garbage_truck:n03417042",
    "gasmask:n03424325",
    "gas_pump:n03425413",
    "goblet:n03443371",
    "go-kart:n03444034",
    "golf_ball:n03445777",
    "golfcart:n03445924",
    "gondola:n03447447",
    "gong:n03447721",
    "gown:n03450230",
    "grand_piano:n03452741",
    "greenhouse:n03457902",
    "grille:n03459775",
    "grocery_store:n03461385",
    "guillotine:n03467068",
    "hair_slide:n03476684",
    "hair_spray:n03476991",
    "half_track:n03478589",
    "hammer:n03481172",
    "hamper:n03482405",
    "hand_blower:n03483316",
    "hand-held_computer:n03485407",
    "handkerchief:n03485794",
    "hard_disc:n03492542",
    "harmonica:n03494278",
    "harp:n03495258",
    "harvester:n03496892",
    "hatchet:n03498962",
    "holster:n03527444",
    "home_theater:n03529860",
    "honeycomb:n03530642",
    "hook:n03532672",
    "hoopskirt:n03534580",
    "horizontal_bar:n03535780",
    "horse_cart:n03538406",
    "hourglass:n03544143",
    "iPod:n03584254",
    "iron:n03584829",
    "jack-o'-lantern:n03590841",
    "jean:n03594734",
    "jeep:n03594945",
    "jersey:n03595614",
    "jigsaw_puzzle:n03598930",
    "jinrikisha:n03599486",
    "joystick:n03602883",
    "kimono:n03617480",
    "knee_pad:n03623198",
    "knot:n03627232",
    "lab_coat:n03630383",
    "ladle:n03633091",
    "lampshade:n03637318",
    "laptop:n03642806",
    "lawn_mower:n03649909",
    "lens_cap:n03657121",
    "letter_opener:n03658185",
    "library:n03661043",
    "lifeboat:n03662601",
    "lighter:n03666591",
    "limousine:n03670208",
    "liner:n03673027",
    "lipstick:n03676483",
    "Loafer:n03680355",
    "lotion:n03690938",
    "loudspeaker:n03691459",
    "loupe:n03692522",
    "lumbermill:n03697007",
    "magnetic_compass:n03706229",
    "mailbag:n03709823",
    "mailbox:n03710193",
    "maillot:n03710637",
    "maillot:n03710721",
    "manhole_cover:n03717622",
    "maraca:n03720891",
    "marimba:n03721384",
    "mask:n03724870",
    "matchstick:n03729826",
    "maypole:n03733131",
    "maze:n03733281",
    "measuring_cup:n03733805",
    "medicine_chest:n03742115",
    "megalith:n03743016",
    "microphone:n03759954",
    "microwave:n03761084",
    "military_uniform:n03763968",
    "milk_can:n03764736",
    "minibus:n03769881",
    "miniskirt:n03770439",
    "minivan:n03770679",
    "missile:n03773504",
    "mitten:n03775071",
    "mixing_bowl:n03775546",
    "mobile_home:n03776460",
    "Model_T:n03777568",
    "modem:n03777754",
    "monastery:n03781244",
    "monitor:n03782006",
    "moped:n03785016",
    "mortar:n03786901",
    "mortarboard:n03787032",
    "mosque:n03788195",
    "mosquito_net:n03788365",
    "motor_scooter:n03791053",
    "mountain_bike:n03792782",
    "mountain_tent:n03792972",
    "mouse:n03793489",
    "mousetrap:n03794056",
    "moving_van:n03796401",
    "muzzle:n03803284",
    "nail:n03804744",
    "neck_brace:n03814639",
    "necklace:n03814906",
    "nipple:n03825788",
    "notebook:n03832673",
    "obelisk:n03837869",
    "oboe:n03838899",
    "ocarina:n03840681",
    "odometer:n03841143",
    "oil_filter:n03843555",
    "organ:n03854065",
    "oscilloscope:n03857828",
    "overskirt:n03866082",
    "oxcart:n03868242",
    "oxygen_mask:n03868863",
    "packet:n03871628",
    "paddle:n03873416",
    "paddlewheel:n03874293",
    "padlock:n03874599",
    "paintbrush:n03876231",
    "pajama:n03877472",
    "palace:n03877845",
    "panpipe:n03884397",
    "paper_towel:n03887697",
    "parachute:n03888257",
    "parallel_bars:n03888605",
    "park_bench:n03891251",
    "parking_meter:n03891332",
    "passenger_car:n03895866",
    "patio:n03899768",
    "pay-phone:n03902125",
    "pedestal:n03903868",
    "pencil_box:n03908618",
    "pencil_sharpener:n03908714",
    "perfume:n03916031",
    "Petri_dish:n03920288",
    "photocopier:n03924679",
    "pick:n03929660",
    "pickelhaube:n03929855",
    "picket_fence:n03930313",
    "pickup:n03930630",
    "pier:n03933933",
    "piggy_bank:n03935335",
    "pill_bottle:n03937543",
    "pillow:n03938244",
    "ping-pong_ball:n03942813",
    "pinwheel:n03944341",
    "pirate:n03947888",
    "pitcher:n03950228",
    "plane:n03954731",
    "planetarium:n03956157",
    "plastic_bag:n03958227",
    "plate_rack:n03961711",
    "plow:n03967562",
    "plunger:n03970156",
    "Polaroid_camera:n03976467",
    "pole:n03976657",
    "police_van:n03977966",
    "poncho:n03980874",
    "pool_table:n03982430",
    "pop_bottle:n03983396",
    "pot:n03991062",
    "potter's_wheel:n03992509",
    "power_drill:n03995372",
    "prayer_rug:n03998194",
    "printer:n04004767",
    "prison:n04005630",
    "projectile:n04008634",
    "projector:n04009552",
    "puck:n04019541",
    "punching_bag:n04023962",
    "purse:n04026417",
    "quill:n04033901",
    "quilt:n04033995",
    "racer:n04037443",
    "racket:n04039381",
    "radiator:n04040759",
    "radio:n04041544",
    "radio_telescope:n04044716",
    "rain_barrel:n04049303",
    "recreational_vehicle:n04065272",
    "reel:n04067472",
    "reflex_camera:n04069434",
    "refrigerator:n04070727",
    "remote_control:n04074963",
    "restaurant:n04081281",
    "revolver:n04086273",
    "rifle:n04090263",
    "rocking_chair:n04099969",
    "rotisserie:n04111531",
    "rubber_eraser:n04116512",
    "rugby_ball:n04118538",
    "rule:n04118776",
    "running_shoe:n04120489",
    "safe:n04125021",
    "safety_pin:n04127249",
    "saltshaker:n04131690",
    "sandal:n04133789",
    "sarong:n04136333",
    "sax:n04141076",
    "scabbard:n04141327",
    "scale:n04141975",
    "school_bus:n04146614",
    "schooner:n04147183",
    "scoreboard:n04149813",
    "screen:n04152593",
    "screw:n04153751",
    "screwdriver:n04154565",
    "seat_belt:n04162706",
    "sewing_machine:n04179913",
    "shield:n04192698",
    "shoe_shop:n04200800",
    "shoji:n04201297",
    "shopping_basket:n04204238",
    "shopping_cart:n04204347",
    "shovel:n04208210",
    "shower_cap:n04209133",
    "shower_curtain:n04209239",
    "ski:n04228054",
    "ski_mask:n04229816",
    "sleeping_bag:n04235860",
    "slide_rule:n04238763",
    "sliding_door:n04239074",
    "slot:n04243546",
    "snorkel:n04251144",
    "snowmobile:n04252077",
    "snowplow:n04252225",
    "soap_dispenser:n04254120",
    "soccer_ball:n04254680",
    "sock:n04254777",
    "solar_dish:n04258138",
    "sombrero:n04259630",
    "soup_bowl:n04263257",
    "space_bar:n04264628",
    "space_heater:n04265275",
    "space_shuttle:n04266014",
    "spatula:n04270147",
    "speedboat:n04273569",
    "spider_web:n04275548",
    "spindle:n04277352",
    "sports_car:n04285008",
    "spotlight:n04286575",
    "stage:n04296562",
    "steam_locomotive:n04310018",
    "steel_arch_bridge:n04311004",
    "steel_drum:n04311174",
    "stethoscope:n04317175",
    "stole:n04325704",
    "stone_wall:n04326547",
    "stopwatch:n04328186",
    "stove:n04330267",
    "strainer:n04332243",
    "streetcar:n04335435",
    "stretcher:n04336792",
    "studio_couch:n04344873",
    "stupa:n04346328",
    "submarine:n04347754",
    "suit:n04350905",
    "sundial:n04355338",
    "sunglass:n04355933",
    "sunglasses:n04356056",
    "sunscreen:n04357314",
    "suspension_bridge:n04366367",
    "swab:n04367480",
    "sweatshirt:n04370456",
    "swimming_trunks:n04371430",
    "swing:n04371774",
    "switch:n04372370",
    "syringe:n04376876",
    "table_lamp:n04380533",
    "tank:n04389033",
    "tape_player:n04392985",
    "teapot:n04398044",
    "teddy:n04399382",
    "television:n04404412",
    "tennis_ball:n04409515",
    "thatch:n04417672",
    "theater_curtain:n04418357",
    "thimble:n04423845",
    "thresher:n04428191",
    "throne:n04429376",
    "tile_roof:n04435653",
    "toaster:n04442312",
    "tobacco_shop:n04443257",
    "toilet_seat:n04447861",
    "torch:n04456115",
    "totem_pole:n04458633",
    "tow_truck:n04461696",
    "toyshop:n04462240",
    "tractor:n04465501",
    "trailer_truck:n04467665",
    "tray:n04476259",
    "trench_coat:n04479046",
    "tricycle:n04482393",
    "trimaran:n04483307",
    "tripod:n04485082",
    "triumphal_arch:n04486054",
    "trolleybus:n04487081",
    "trombone:n04487394",
    "tub:n04493381",
    "turnstile:n04501370",
    "typewriter_keyboard:n04505470",
    "umbrella:n04507155",
    "unicycle:n04509417",
    "upright:n04515003",
    "vacuum:n04517823",
    "vase:n04522168",
    "vault:n04523525",
    "velvet:n04525038",
    "vending_machine:n04525305",
    "vestment:n04532106",
    "viaduct:n04532670",
    "violin:n04536866",
    "volleyball:n04540053",
    "waffle_iron:n04542943",
    "wall_clock:n04548280",
    "wallet:n04548362",
    "wardrobe:n04550184",
    "warplane:n04552348",
    "washbasin:n04553703",
    "washer:n04554684",
    "water_bottle:n04557648",
    "water_jug:n04560804",
    "water_tower:n04562935",
    "whiskey_jug:n04579145",
    "whistle:n04579432",
    "wig:n04584207",
    "window_screen:n04589890",
    "window_shade:n04590129",
    "Windsor_tie:n04591157",
    "wine_bottle:n04591713",
    "wing:n04592741",
    "wok:n04596742",
    "wooden_spoon:n04597913",
    "wool:n04599235",
    "worm_fence:n04604644",
    "wreck:n04606251",
    "yawl:n04612504",
    "yurt:n04613696",
    "web_site:n06359193",
    "comic_book:n06596364",
    "crossword_puzzle:n06785654",
    "street_sign:n06794110",
    "traffic_light:n06874185",
    "book_jacket:n07248320",
    "menu:n07565083",
    "plate:n07579787",
    "guacamole:n07583066",
    "consomme:n07584110",
    "hot_pot:n07590611",
    "trifle:n07613480",
    "ice_cream:n07614500",
    "ice_lolly:n07615774",
    "French_loaf:n07684084",
    "bagel:n07693725",
    "pretzel:n07695742",
    "cheeseburger:n07697313",
    "hotdog:n07697537",
    "mashed_potato:n07711569",
    "head_cabbage:n07714571",
    "broccoli:n07714990",
    "cauliflower:n07715103",
    "zucchini:n07716358",
    "spaghetti_squash:n07716906",
    "acorn_squash:n07717410",
    "butternut_squash:n07717556",
    "cucumber:n07718472",
    "artichoke:n07718747",
    "bell_pepper:n07720875",
    "cardoon:n07730033",
    "mushroom:n07734744",
    "Granny_Smith:n07742313",
    "strawberry:n07745940",
    "orange:n07747607",
    "lemon:n07749582",
    "fig:n07753113",
    "pineapple:n07753275",
    "banana:n07753592",
    "jackfruit:n07754684",
    "custard_apple:n07760859",
    "pomegranate:n07768694",
    "hay:n07802026",
    "carbonara:n07831146",
    "chocolate_sauce:n07836838",
    "dough:n07860988",
    "meat_loaf:n07871810",
    "pizza:n07873807",
    "potpie:n07875152",
    "burrito:n07880968",
    "red_wine:n07892512",
    "espresso:n07920052",
    "cup:n07930864",
    "eggnog:n07932039",
    "alp:n09193705",
    "bubble:n09229709",
    "cliff:n09246464",
    "coral_reef:n09256479",
    "geyser:n09288635",
    "lakeside:n09332890",
    "promontory:n09399592",
    "sandbar:n09421951",
    "seashore:n09428293",
    "valley:n09468604",
    "volcano:n09472597",
    "ballplayer:n09835506",
    "groom:n10148035",
    "scuba_diver:n10565667",
    "rapeseed:n11879895",
    "daisy:n11939491",
    "yellow_lady's_slipper:n12057211",
    "corn:n12144580",
    "acorn:n12267677",
    "hip:n12620546",
    "buckeye:n12768682",
    "coral_fungus:n12985857",
    "agaric:n12998815",
    "gyromitra:n13037406",
    "stinkhorn:n13040303",
    "earthstar:n13044778",
    "hen-of-the-woods:n13052670",
    "bolete:n13054560",
    "ear:n13133613",
    "toilet_tissue:n15075141",
]