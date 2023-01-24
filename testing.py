class User:
    def __init__(self, user: str, password: str) -> None:
        self.user= user
        self.password = password
    
    def to_dict(self) -> dict[str, str]:
        return {
            'username': self.user,
            'password': self.password
        } 

def test(user: User) -> dict[str, str]:
    return user.to_dict()

username = 'Ton'
password = 'Borrell'

user = User(username, password)
print(test(user))