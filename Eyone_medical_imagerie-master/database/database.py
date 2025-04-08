from motor.motor_asyncio import AsyncIOMotorClient


#mongo_client = AsyncIOMotorClient("mongodb+srv://northatech:5kCOjn4SS6HP3N0F@atlascluster.our8m0y.mongodb.net/")
mongo_client = AsyncIOMotorClient("mongodb://localhost:27017")
db = mongo_client['eyone']

def rdvs_func():
    rdvs = db['rdvs']
    return rdvs


def histories():
    histories = db['histories']
    return histories

def historieswo():
    histories = db['historieswo']
    return histories
