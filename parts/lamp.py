from phue import Bridge


class Lamp:
    def __init__(self, ip='192.168.0.151'):
        self.b = Bridge(ip)
        self.b.connect()
        self.b.get_api()

        self.lamp_id = 2

    def turn_on(self):
        self.b.set_light(self.lamp_id, 'on', True)

    def turn_off(self):
        self.b.set_light(self.lamp_id, 'on', False)

    def toggle(self):
        self.b.set_light(self.lamp_id, 'on', not self.b.get_light(self.lamp_id, "on"))

    def change_brightness(self, brightness):
        self.b.set_light(self.lamp_id, 'bri', brightness)

    def change_color_temp(self, temp):
        self.b.set_light(self.lamp_id, 'ct', int(250 + (454 - 250) / 100 * temp))
