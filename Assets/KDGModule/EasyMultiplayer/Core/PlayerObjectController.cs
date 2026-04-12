using Mirror;
using UnityEngine;
using Steamworks;

public class PlayerObjectController : NetworkBehaviour
{
    #region MEMBER_BASE
    [SyncVar]
    public int ConnectionID;
    [SyncVar]
    public int PlayerIdNumber;
    [SyncVar]
    public ulong PlayerSteamID;
    #endregion

    private CustomNetworkManager manager;
    private CustomNetworkManager Manager
    {
        get
        {
            if (manager != null)
            {
                return manager;
            }
            return manager = CustomNetworkManager.singleton as CustomNetworkManager;
        }
    }
    public override void OnStartClient()
    {
        Manager.GamePlayers.Add(this);
    }

    public override void OnStopClient()
    {
        Manager.GamePlayers.Remove(this);
    }

    private void Start()
    {
        DontDestroyOnLoad(this.gameObject);
    }
}
