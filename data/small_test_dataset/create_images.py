
from PIL import Image, ImageDraw
from pathlib import Path

# Create images folder
IMAGE_DIR = Path(__file__).parent / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

WIDTH, HEIGHT = 640, 420


def create_beach():
    img = Image.new("RGB", (WIDTH, HEIGHT), (140, 205, 240))
    draw = ImageDraw.Draw(img)

    draw.ellipse((500, 35, 575, 110), fill=(255, 215, 70))
    draw.rectangle((0, 225, WIDTH, 330), fill=(35, 145, 205))
    draw.polygon(
        [(0, 285), (180, 235), (370, 275),
         (640, 230), (640, 420), (0, 420)],
        fill=(245, 215, 150)
    )
    draw.text((20, 20), "BEACH SCENE", fill="black")
    img.save(IMAGE_DIR / "beach.png")


def create_soccer():
    img = Image.new("RGB", (WIDTH, HEIGHT), (220, 240, 250))
    draw = ImageDraw.Draw(img)

    draw.rectangle((0, 100, WIDTH, HEIGHT), fill=(55, 155, 75))
    draw.rectangle((210, 170, 430, 360), outline="white", width=5)
    draw.ellipse((275, 205, 365, 295), outline="white", width=4)
    draw.ellipse((300, 265, 340, 305), fill="white", outline="black")

    for x, y, color in [
        (150, 220, "blue"),
        (475, 240, "red"),
        (245, 330, "blue"),
        (400, 330, "red")
    ]:
        draw.ellipse((x-18, y-58, x+18, y-22), fill="peachpuff")
        draw.rectangle((x-20, y-20, x+20, y+40), fill=color)

    draw.text((20, 20), "SOCCER MATCH", fill="black")
    img.save(IMAGE_DIR / "soccer.png")


def create_mountain():
    img = Image.new("RGB", (WIDTH, HEIGHT), (150, 205, 240))
    draw = ImageDraw.Draw(img)

    draw.ellipse((500, 35, 565, 100), fill=(255, 220, 90))
    draw.polygon(
        [(20, 360), (210, 85), (390, 360)],
        fill=(95, 115, 135)
    )
    draw.polygon(
        [(260, 360), (470, 65), (650, 360)],
        fill=(105, 125, 145)
    )
    draw.polygon(
        [(210, 85), (160, 165), (210, 145), (255, 175)],
        fill="white"
    )
    draw.polygon(
        [(470, 65), (420, 155), (470, 130), (520, 165)],
        fill="white"
    )
    draw.rectangle((0, 360, WIDTH, HEIGHT), fill=(70, 135, 85))

    draw.text((20, 20), "SNOWY MOUNTAINS", fill="black")
    img.save(IMAGE_DIR / "mountain.png")


def create_dog():
    img = Image.new("RGB", (WIDTH, HEIGHT), (180, 220, 245))
    draw = ImageDraw.Draw(img)

    draw.rectangle((0, 260, WIDTH, HEIGHT), fill=(80, 165, 80))

    draw.ellipse((250, 150, 420, 310), fill=(180, 120, 65))
    draw.ellipse((225, 135, 300, 245), fill=(135, 85, 45))
    draw.ellipse((370, 135, 445, 245), fill=(135, 85, 45))

    draw.ellipse((300, 195, 320, 215), fill="black")
    draw.ellipse((360, 195, 380, 215), fill="black")
    draw.ellipse((325, 230, 355, 252), fill="black")

    draw.line((285, 290, 275, 350), fill=(80, 50, 25), width=10)
    draw.line((385, 290, 395, 350), fill=(80, 50, 25), width=10)

    draw.text((20, 20), "DOG IN A PARK", fill="black")
    img.save(IMAGE_DIR / "dog.png")


def create_car():
    img = Image.new("RGB", (WIDTH, HEIGHT), (185, 220, 245))
    draw = ImageDraw.Draw(img)

    draw.rectangle((0, 290, WIDTH, HEIGHT), fill=(95, 105, 110))

    draw.rounded_rectangle(
        (150, 190, 500, 315),
        radius=35,
        fill=(205, 45, 45)
    )

    draw.polygon(
        [(220, 190), (285, 120), (410, 120), (465, 190)],
        fill=(190, 225, 245)
    )

    draw.ellipse((205, 280, 290, 365), fill=(35, 35, 35))
    draw.ellipse((385, 280, 470, 365), fill=(35, 35, 35))

    draw.text((20, 20), "RED CAR", fill="black")
    img.save(IMAGE_DIR / "car.png")


if __name__ == "__main__":
    create_beach()
    create_soccer()
    create_mountain()
    create_dog()
    create_car()

    print("Successfully created 5 sample images!")

    for image in IMAGE_DIR.iterdir():
        if image.suffix == ".png":
            print(image.name)
