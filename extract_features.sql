
DROP TABLE IF EXISTS bat_final;
DROP TABLE IF EXISTS bat_temp;
DROP TABLE IF EXISTS field_final;
DROP TABLE IF EXISTS field_temp;
DROP TABLE IF EXISTS pitch_final;
DROP TABLE IF EXISTS pitch_temp;
DROP TABLE IF EXISTS final_set;
DROP TABLE IF EXISTS hof_final;
DROP TABLE IF EXISTS nominated;

# Select and aggregate important batting stats
CREATE TABLE bat_temp
   AS SELECT playerID, yearID, G, AB, R, H, 2B, 3B, HR, RBI, SO
   FROM Batting;
INSERT INTO bat_temp
   SELECT playerID, yearID, G, AB, R, H, 2B, 3B, HR, RBI, SO
   FROM BattingPost;
CREATE TABLE bat_final
   AS SELECT playerID,
       count(DISTINCT yearID) AS yearsActiveBatting,
       sum(G) as GBatting,
       sum(AB) as AB,
       sum(R) as R,
       sum(H) as H,
       sum(2B) as 2B,
       sum(3B) as 3B,
       sum(HR) as HR,
       sum(RBI) as RBI,
       sum(SO) as SO
   FROM bat_temp
   GROUP BY playerID;
DROP TABLE bat_temp;
DESCRIBE bat_final;

# Select and aggregate important fielding stats
CREATE TABLE field_temp
   AS SELECT playerID, yearID, G, PO, A, E, DP, PB, SB, CS
   FROM Fielding;
INSERT INTO field_temp
   SELECT playerID, yearID, G, PO, A, E, DP, PB, SB, CS
   FROM FieldingOFsplit;
INSERT INTO field_temp
   SELECT playerID, yearID, G, PO, A, E, DP, PB, SB, CS
   FROM FieldingPost;
CREATE TABLE field_final
   AS SELECT playerID,
       count(DISTINCT yearID) AS yearsActiveFielding,
       sum(G) as GFielding,
       sum(PO) as PO,
       sum(A) as A,
       sum(E) as E,
       sum(DP) as DP,
       sum(PB) as PB,
       sum(SB) as SB,
       sum(CS) as CS
   FROM field_temp
   GROUP BY playerID;
DROP TABLE field_temp;
DESCRIBE field_final;

# Select and aggregate important pitching stats 
CREATE TABLE pitch_temp
    AS SELECT playerID, yearID, W, L, G, SHO, SV, IPOuts, H, ER, HR, BB, SO, BFP
    FROM Pitching;
INSERT INTO pitch_temp
    SELECT playerID, yearID, W, L, G, SHO, SV, IPOuts, H, ER, HR, BB, SO, BFP
    FROM PitchingPost;
CREATE TABLE pitch_final
    AS SELECT playerID,
        count(DISTINCT yearID) as yearsActivePitching,
        sum(W) as W,
        sum(L) as L,
        sum(G) as GPitching,
        sum(SHO) as SHO,
        sum(SV) as SV,
        sum(IPOuts) as IPOuts,
        sum(H) as HA,         # Hits Against
        sum(ER) as ER,
        sum(HR) as HRA,       # Home Runs Against
        sum(BB) as BB,
        sum(SO) as SOF,       # Strike Outs For
        sum(BFP) as BFP
    FROM pitch_temp
    GROUP BY playerID;
DROP TABLE pitch_temp;
DESCRIBE pitch_final;

# Select and aggregate the hall of fame tables to include a unique list of all 
# playerIDs that have either been nominated or elected.
CREATE TABLE hof_final 
    AS SELECT *
    FROM (
        SELECT playerID, inducted
            FROM HallOfFame 
            WHERE category="Player"
    )
    AS hof
    WHERE inducted="Y";
CREATE TABLE nominated
    AS SELECT DISTINCT playerID, inducted
    FROM (
        SELECT playerID, inducted
            FROM HallOfFame 
            WHERE category="Player"
    )
    AS hof
    WHERE playerID NOT IN (
        SELECT playerID 
            FROM hof_final
    );
INSERT INTO hof_final 
    SELECT * from nominated;
DROP TABLE nominated;
DESCRIBE hof_final;

# Total number of features: 33 (years counted seperately, debut not included till below select)
# Total number of columns: 34
CREATE TABLE final_set AS SELECT
  bf.*,
  ff.yearsActiveFielding, ff.GFielding, ff.PO, ff.A, ff.E, ff.DP, ff.PB, ff.SB, ff.CS,
  pf.yearsActivePitching, pf.W, pf.L, pf.GPitching, pf.SHO, pf.SV, pf.IPOuts, pf.HA, pf.ER, pf.HRA, pf.BB, pf.SOF, pf.BFP,
  m.debut,
  h.inducted
  FROM field_final ff
    LEFT JOIN pitch_final pf
      ON ff.playerID=pf.playerID
    LEFT JOIN bat_final bf
      ON ff.playerID=bf.playerID
    LEFT JOIN Master m
      ON ff.playerID=m.playerID
		   AND pf.playerID=m.playerID
         AND bf.playerID=m.playerID
    RIGHT JOIN hof_final h
      ON ff.playerID=h.playerID;
DESCRIBE final_set;

Output features to CSV 
SELECT "playerID","yearsActiveBatting","GBatting","AB","R","H","2B","3B","HR","RBI","SO",
"yearsActiveFielding","GFielding","PO","A","E","DP","PB","SB","CS","yearsActivePitching",
"W","L","GPitching","SHO","SV","IPOuts","HA","ER","HRA","BB","SOF","BFP","debut","inducted"
UNION ALL
SELECT * FROM final_set
INTO OUTFILE '/home/mackenzieskyewilson/src/halloffamepredictor/output_files/extracted_features.csv' # This path based on 'secure_file_priv' SQL variable
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n';