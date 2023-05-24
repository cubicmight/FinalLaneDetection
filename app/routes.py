import glob
import json
import os.path
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import cvzone
from flask import render_template, flash, redirect, url_for, Response, request, send_from_directory
from flask_login import current_user, login_user, logout_user, login_required
from werkzeug.urls import url_parse
from app import app, db
from app.forms import LoginForm, RegistrationForm
from app.models import User, UserLogData
import math
import cv2      # For video/image processing
from moviepy.editor import VideoFileClip    # For image recording and saving
from lane_funcs import process_frame, arrow_detection_left, arrow_detection_right, region_of_interest



current_direction_image = None
camera = None

with app.app_context():
    UserLogData.query.delete()
    db.session.commit()
    basedir = os.path.abspath(os.path.dirname(__file__))

    image_file_name = "direction_arrow.png"
    full_image_path = os.path.join(basedir, 'static', image_file_name)
    if os.path.exists(full_image_path):
        # validate the image
        current_direction_image = cv2.imread(full_image_path, cv2.IMREAD_UNCHANGED)
    else:
        print("cannot find image")


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')


@app.route('/')
@app.route('/index')
@login_required
def index():
    return render_template("index.html")


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        details = {
            "user": current_user.username.data,
            "success": True,
            "log-in": current_user + "has logged in"
        }

        r = json.dumps(details)
        data = UserLogData(data=r)
        db.session.add(data)
        db.session.commit()
        return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        details = {
            "user": form.username.data,
            "success": True,
            "log-in": form.username.data + "has logged in"
        }

        r = json.dumps(details)
        data = UserLogData(data=r)
        db.session.add(data)
        db.session.commit()
        if user is None or not user.check_password(form.password.data):
            flash('Invalid username or password')
            return redirect(url_for('login'))
        login_user(user, remember=form.remember_me.data)
        next_page = request.args.get('next')
        if not next_page or url_parse(next_page).netloc != '':
            next_page = url_for('index')
        return redirect(next_page)
    return render_template('login.html', title='Sign In', form=form)




@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))


@app.route('/logs')
def logs():
    data = db.session.query(UserLogData.data).all()
    row = []
    for d in data:
        row.append(d[0])
    return row


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Congratulations, you are now a registered user!')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)

def gen_frame():
    """Video streaming generator function."""
    global camera
    if not camera:
        basedir = os.path.abspath(os.path.dirname(__file__))

        image_file_name = "../videos/IMG_8417.mp4"
        full_image_path = os.path.join(basedir, image_file_name)
        if not os.path.exists(full_image_path):
            print("cannot find image")
            exit(-1)

        cap = cv2.VideoCapture(full_image_path)
        # fixed_cap = format_video(cap)

    else:
        cap = camera
    while cap:
        (grabbed, frame) = cap.read()
        if grabbed:
            global current_direction_image
            imgResult = cvzone.overlayPNG(process_frame(frame), current_direction_image, (900, 400))#1250, 660
            ret, buffer = cv2.imencode('.jpg', imgResult)
            if ret:
                convert = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + convert + b'\r\n')
                # concatenate frame one by one and show result
    cap.release()


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
