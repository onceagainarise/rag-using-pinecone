from werkzeug.security import generate_password_hash

password = 'dholakpur'
hashed_password = generate_password_hash(password)
print(hashed_password)
