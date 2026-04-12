using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Mirror;
using Steamworks;
using UnityEngine.UI;
using System;

public class SteamLobby : MonoBehaviour
{
    public static SteamLobby Instance;

    protected Callback<LobbyCreated_t> LobbyCreated;
    protected Callback<GameLobbyJoinRequested_t> JoinRequest;
    protected Callback<LobbyEnter_t> LobbyEntered;

    public ulong CurrentLobbyID;
    private const string HostAddressKey = "CustomHostAddress";
    private CustomNetworkManager manager;

    private void Start()
    {
        if (!SteamManager.Initialized)
            return;

        if (Instance == null)
            Instance = this;

        manager = GetComponent<CustomNetworkManager>();

        LobbyCreated = Callback<LobbyCreated_t>.Create(OnLobbyCreated);
        JoinRequest = Callback<GameLobbyJoinRequested_t>.Create(OnJoinRequest);
        LobbyEntered = Callback<LobbyEnter_t>.Create(OnLobbyEntered);
    }

    //버튼 등으로 로비 호스트할 때 실행
    public void HostLobby()
    {
        SteamMatchmaking.CreateLobby(ELobbyType.k_ELobbyTypeFriendsOnly, manager.maxConnections);
    }

    //로비가 생성되었을 때 콜백
    private void OnLobbyCreated(LobbyCreated_t callback)
    {
        if (callback.m_eResult != EResult.k_EResultOK)
            return;

        Debug.Log("로비 생성 성공");

        manager.StartHost();

        SteamMatchmaking.SetLobbyData(new CSteamID(callback.m_ulSteamIDLobby), HostAddressKey, SteamUser.GetSteamID().ToString());
        SteamMatchmaking.SetLobbyData(new CSteamID(callback.m_ulSteamIDLobby), "name", SteamFriends.GetPersonaName().ToString() + "'s Lobby");

    }

    //로비 참여 시 콜백
    private void OnJoinRequest(GameLobbyJoinRequested_t callback)
    {
        Debug.Log("로비 참여 요청");
        SteamMatchmaking.JoinLobby(callback.m_steamIDLobby);
    }

    //로비 입장 시 
    private void OnLobbyEntered(LobbyEnter_t callback)
    {

        CurrentLobbyID = callback.m_ulSteamIDLobby;

        if (NetworkServer.active)
            return;

        manager.networkAddress = SteamMatchmaking.GetLobbyData(new CSteamID(callback.m_ulSteamIDLobby), HostAddressKey);

        manager.StartClient();
    }

}