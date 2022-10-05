
CREATE DATABASE SportsClub;
CREATE TABLE Players (playerID INT, playerName VARCHAR(50), age INT, PRIMARY KEY(playerID));
INSERT INTO Players (playerID, playerName, age) VALUES (1, "Jack", 25);

INSERT INTO Players (playerID, playerName, age) VALUES (2, "Karl", 20);
INSERT INTO Players (playerID, playerName, age) VALUES (3, "Mark", 21);
INSERT INTO Players (playerID, playerName, age) VALUES (4, "Andrew", 22);

SELECT playerName FROM Players WHERE playerID = 2;