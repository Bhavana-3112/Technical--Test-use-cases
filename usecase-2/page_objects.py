class LoginPage:
    def __init__(self, page):
        self.page = page

    def goto(self):
        self.page.goto('https://example.com/login')

    def login(self, username, password):
        self.page.fill('#username', username)
        self.page.fill('#password', password)
        self.page.click('#loginButton')
