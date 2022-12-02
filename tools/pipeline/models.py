from database import db
from flask_login import UserMixin

class AnomalyDatabase(db.Model):
    """
    Params:
        id<int>: Index of row in database
        image_upload<str>: Path of image upload
        anomaly_score<float>: Anomaly score predict
        target<str>: Target label predict
        heatmap_predict<str>: Path of heatmap predict
        mask_predict<str>: Path of mask predict
        segment_predict<str>: Path of segment predict
    """
    __tablename__ = 'anomaly'
    __table_args__ = {'extend_existing': True}
    id = db.Column("id", db.Integer, primary_key=True)
    image_upload = db.Column(db.String(100))
    anomaly_score = db.Column(db.Float)
    target = db.Column(db.String(50))
    heatmap_predict = db.Column(db.String(100))
    mask_predict = db.Column(db.String(100))
    segment_predict = db.Column(db.String(100))

    def __repr__(self):
        return f"{self.id}-{self.image_upload}-{self.heatmap_predict}-{self.mask_predict}-{self.segment_predict}"

class User(UserMixin, db.Model):
    """
    Params:
        id<int>: Index of row in database
        password<str>: User password
        name<str>: User name
    """
    __tablename__ = 'user'
    __table_args__ = {'extend_existing': True}
    id = db.Column("user_id", db.Integer, primary_key=True)
    password = db.Column(db.String(100))
    name = db.Column(db.String(100), unique=True)
