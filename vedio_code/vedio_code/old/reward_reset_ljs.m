function  rewd = reward_reset_ljs(rewd)

rewd.numdot = 0;
rewd.numgoast = 0;
rewd.numeneg = 0;
rewd.magdot = 2;
rewd.maggoast = 8;
rewd.mageneg = 4;
rewd.probdot = 2;
rewd.probgoast = 1;
rewd.probeneg = 1;
% ============ fruit ============
rewd.magcherry = 3;
rewd.magstrawberry = 5;
rewd.magorange = 8;
rewd.magapple = 12;
rewd.magmelon = 17;

end
