from flask import Flask, request, jsonify, make_response
import jwt
from functools import wraps
from model import User
from config import SECRET_KEY


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        if 'x-access-token' in request.headers:
            token = request.headers['x-access-token']
            
        if not token:
            return jsonify({'message': 'Token is missing !!'}), 401

        try:
            data = jwt.decode(token, SECRET_KEY)
            current_user = User.query\
                .filter_by(username=data['username'])\
                .first()
        except:
            return jsonify({
                'message': 'Token is invalid !!'
            }), 401

        return f(current_user, *args, **kwargs)

    return decorated
