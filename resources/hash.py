import bcrypt
password_to_hash = b'prueba456' # La contraseña como bytes
salt = bcrypt.gensalt()
hashed_password = bcrypt.hashpw(password_to_hash, salt)
print(hashed_password.decode('utf-8')) # Este es el hash que debes guardar