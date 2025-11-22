import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class ModelManager:
    """Manages ResNet18 model loading and inference with GPU support."""

    _instance = None
    _model = None
    _device = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is None:
            self._load_model()

    def _load_model(self):
        """Load ResNet18 with ImageNet pretrained weights."""
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading ResNet18 on device: {self._device}")

        # Load pretrained ResNet18
        self._model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self._model = self._model.to(self._device)
        self._model.eval()

        print("ResNet18 model loaded successfully!")

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def device(self) -> torch.device:
        return self._device

    def predict(self, image_tensor: torch.Tensor) -> tuple[int, float, str]:
        """
        Make prediction on input image tensor.

        Args:
            image_tensor: Preprocessed image tensor (1, 3, 224, 224)

        Returns:
            Tuple of (class_index, confidence, class_name)
        """
        with torch.no_grad():
            output = self._model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        class_idx = predicted.item()
        conf = confidence.item()
        class_name = self._get_class_name(class_idx)

        return class_idx, conf, class_name

    def _get_class_name(self, class_idx: int) -> str:
        """Get ImageNet class name from index."""
        # ImageNet class names (subset for display)
        imagenet_classes = self._get_imagenet_classes()
        return imagenet_classes.get(class_idx, f"class_{class_idx}")

    def _get_imagenet_classes(self) -> dict:
        """Return ImageNet class mapping."""
        # This is a subset of ImageNet classes for demonstration
        # In production, you'd load the full 1000 class names
        return {
            0: "tench", 1: "goldfish", 2: "great_white_shark", 3: "tiger_shark",
            4: "hammerhead", 5: "electric_ray", 6: "stingray", 7: "cock",
            8: "hen", 9: "ostrich", 10: "brambling", 11: "goldfinch",
            12: "house_finch", 13: "junco", 14: "indigo_bunting", 15: "robin",
            16: "bulbul", 17: "jay", 18: "magpie", 19: "chickadee",
            20: "water_ouzel", 21: "kite", 22: "bald_eagle", 23: "vulture",
            24: "great_grey_owl", 25: "European_fire_salamander", 26: "common_newt",
            27: "eft", 28: "spotted_salamander", 29: "axolotl", 30: "bullfrog",
            31: "tree_frog", 32: "tailed_frog", 33: "loggerhead", 34: "leatherback_turtle",
            65: "sea_slug", 76: "tarantula", 89: "sulphur_butterfly", 90: "lycaenid",
            99: "goose", 100: "black_swan", 130: "flamingo", 144: "pelican",
            207: "golden_retriever", 208: "Labrador_retriever", 229: "Old_English_sheepdog",
            232: "Border_collie", 243: "bull_mastiff", 244: "Tibetan_mastiff",
            245: "French_bulldog", 249: "malamute", 250: "Siberian_husky",
            258: "Samoyed", 259: "Pomeranian", 263: "Pembroke", 265: "toy_poodle",
            267: "standard_poodle", 281: "tabby", 282: "tiger_cat", 283: "Persian_cat",
            285: "Egyptian_cat", 291: "lion", 292: "tiger", 293: "cheetah",
            294: "brown_bear", 295: "American_black_bear", 296: "ice_bear",
            309: "bee", 310: "ant", 311: "grasshopper", 314: "cockroach",
            327: "starfish", 330: "sea_urchin", 332: "coral_reef", 386: "African_elephant",
            387: "Indian_elephant", 388: "mammoth", 409: "analog_clock", 417: "balloon",
            425: "barn", 430: "basketball", 454: "bottle", 457: "bow_tie",
            466: "bulletproof_vest", 470: "burrito", 487: "cellular_telephone",
            488: "chain", 492: "chest", 497: "church", 504: "coffee_mug",
            508: "computer_keyboard", 531: "digital_watch", 558: "flute",
            566: "freight_car", 569: "fur_coat", 574: "gasmask", 610: "holster",
            614: "hook", 621: "iPod", 629: "laptop", 654: "miniskirt",
            657: "missile", 671: "mountain_bike", 673: "mouse", 675: "moving_van",
            681: "notebook", 701: "parachute", 717: "pickup", 720: "pillow",
            737: "pool_table", 749: "quilt", 751: "racer", 756: "red_wine",
            759: "reflex_camera", 764: "revolver", 768: "rifle", 779: "running_shoe",
            782: "sandal", 795: "ski", 817: "sports_car", 849: "tennis_ball",
            852: "theater_curtain", 859: "toaster", 874: "trolleybus", 879: "umbrella",
            880: "unicycle", 888: "vase", 892: "volcano", 907: "web_site",
            920: "traffic_light", 924: "guacamole", 928: "ice_cream", 931: "bagel",
            932: "pretzel", 933: "cheeseburger", 934: "hot_dog", 936: "mashed_potato",
            937: "broccoli", 938: "cauliflower", 939: "zucchini", 940: "spaghetti_squash",
            945: "bell_pepper", 947: "mushroom", 948: "Granny_Smith", 949: "strawberry",
            950: "orange", 951: "lemon", 952: "fig", 953: "pineapple", 954: "banana",
            955: "jackfruit", 956: "custard_apple", 957: "pomegranate", 963: "pizza",
            964: "burrito", 965: "consomme", 967: "espresso", 968: "bubble",
            985: "daisy", 987: "corn", 988: "acorn", 989: "hip", 991: "buckeye",
            992: "coral_fungus", 993: "agaric", 994: "gyromitra", 997: "bolete"
        }
