-- Face Recognition Evaluation schema - 6 tables, all prefixed Eval*.
-- Completely isolated from production DPP / CDPMC / CDP2000 DBs.

IF OBJECT_ID('dbo.EvalManualJudgment', 'U') IS NOT NULL DROP TABLE dbo.EvalManualJudgment;
IF OBJECT_ID('dbo.EvalEmbedding',       'U') IS NOT NULL DROP TABLE dbo.EvalEmbedding;
IF OBJECT_ID('dbo.EvalFaceDetection',   'U') IS NOT NULL DROP TABLE dbo.EvalFaceDetection;
IF OBJECT_ID('dbo.EvalGamePhoto',       'U') IS NOT NULL DROP TABLE dbo.EvalGamePhoto;
IF OBJECT_ID('dbo.EvalStudioPhoto',     'U') IS NOT NULL DROP TABLE dbo.EvalStudioPhoto;
IF OBJECT_ID('dbo.EvalGames',           'U') IS NOT NULL DROP TABLE dbo.EvalGames;

-- User manually inserts 5 rows (GameNumber + two TeamKeys).
CREATE TABLE dbo.EvalGames (
    GameNumber  INT          NOT NULL PRIMARY KEY,
    TeamKey1    BIGINT       NOT NULL,
    TeamKey2    BIGINT       NOT NULL
);

-- One row per studio JPG discovered under Studio/{TeamKey}/.
CREATE TABLE dbo.EvalStudioPhoto (
    StudioPhotoID  INT IDENTITY(1,1) PRIMARY KEY,
    TeamKey        BIGINT        NOT NULL,
    ImagePath      NVARCHAR(500) NOT NULL
);
CREATE INDEX IX_EvalStudioPhoto_TeamKey ON dbo.EvalStudioPhoto(TeamKey);

-- One row per game JPG discovered under Game/{GameNumber}/.
CREATE TABLE dbo.EvalGamePhoto (
    GamePhotoID  INT IDENTITY(1,1) PRIMARY KEY,
    GameNumber   INT           NOT NULL,
    ImagePath    NVARCHAR(500) NOT NULL
);
CREATE INDEX IX_EvalGamePhoto_GameNumber ON dbo.EvalGamePhoto(GameNumber);

-- One row per MTCNN-detected face (studio or game).
CREATE TABLE dbo.EvalFaceDetection (
    FaceID        INT IDENTITY(1,1) PRIMARY KEY,
    SourceType    CHAR(1)       NOT NULL,   -- 'S' studio, 'G' game
    SourceID      INT           NOT NULL,   -- FK into EvalStudioPhoto or EvalGamePhoto
    BoxX1         INT           NULL,
    BoxY1         INT           NULL,
    BoxX2         INT           NULL,
    BoxY2         INT           NULL,
    Confidence    FLOAT         NULL,       -- MTCNN detection confidence
    FaceCropPath  NVARCHAR(500) NULL        -- aligned crop file for the UI
);
CREATE INDEX IX_EvalFaceDetection_SourceType_SourceID ON dbo.EvalFaceDetection(SourceType, SourceID);

-- One row per (face, model).  5 rows per detected face.
CREATE TABLE dbo.EvalEmbedding (
    EmbeddingID  INT IDENTITY(1,1) PRIMARY KEY,
    FaceID       INT           NOT NULL,
    ModelName    VARCHAR(20)   NOT NULL,    -- facenet|arcface|adaface|magface|lvface
    Embedding    VARBINARY(MAX)    NULL,    -- 512 floats * 4 bytes (LVFace may differ)
    QualityNorm  FLOAT         NULL        -- AdaFace feature-norm only
);
CREATE INDEX IX_EvalEmbedding_FaceID_Model ON dbo.EvalEmbedding(FaceID, ModelName);

-- User clicks in Streamlit, rows appear here.
CREATE TABLE dbo.EvalManualJudgment (
    JudgmentID    INT IDENTITY(1,1) PRIMARY KEY,
    StudioFaceID  INT           NOT NULL,
    GameFaceID    INT           NOT NULL,
    ModelName     VARCHAR(20)   NOT NULL,
    Judgment      CHAR(1)       NOT NULL,  -- 'Y' correct, 'N' incorrect
    Similarity    FLOAT         NOT NULL,
    Threshold     FLOAT         NOT NULL,
    JudgedAt      DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IX_EvalManualJudgment_Model ON dbo.EvalManualJudgment(ModelName);
